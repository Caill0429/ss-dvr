#!/usr/bin/env bash
set -eo pipefail

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

# =======================
# Grid settings
# =======================
K_LIST=(1 2 4 8 16 32 64 128)

# Conda env names
ENV_DINO="dinov3"
ENV_ACE="ace"

# Paths
DINO_DIR="$HOME/PycharmProjects/DINOv3/dinov3"
ACE_DIR="$HOME/PycharmProjects/ss-dvr/ss-dvr"

# DINOv3
DINO_SCRIPT="dinov3_semantic_label_generation-2.py"

# =======================
# Dataset (CLI)
# =======================
DATASET_NAME="${DATASET_NAME:-Cambridge_GreatCourt}"  # default

# Results
RESULTS_FILE="$ACE_DIR/output/semantic-spatial_clustering_results_${DATASET_NAME}.txt"
CELL_W=12  # column width for nice alignment (fits "123.45/67.89")

# =======================
# Timeout & retry policy
# =======================
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-900}"       # test.py 最长允许时间（秒）
RETRAIN_ON_TIMEOUT="${RETRAIN_ON_TIMEOUT:-1}"    # 超时/失败后是否重训再测：1/0
MAX_TEST_RETRIES="${MAX_TEST_RETRIES:-1}"        # 超时/失败后最多重试次数（1=重训+再测1次）
CUDA_DEBUG_SYNC="${CUDA_DEBUG_SYNC:-0}"          # 1=CUDA_LAUNCH_BLOCKING=1 帮助尽快暴露真实错误（会慢）
TORCH_STACKTRACE="${TORCH_STACKTRACE:-1}"        # 1=TORCH_SHOW_CPP_STACKTRACES=1

# =======================
# Start indices (CLI)
# =======================
START_SEM="${START_SEM:-}"   # value in K_LIST, e.g. 32
START_SPA="${START_SPA:-}"   # value in K_LIST, e.g. 16
RESUME=0

# =======================
# NEW: Skip DINOv3 (CLI / ENV)
# =======================
SKIP_DINO="${SKIP_DINO:-0}"  # 1=skip running dinov3 generation
SKIP_DINO_CLEAN="${SKIP_DINO_CLEAN:-0}"  # 1=skip dino but still remove *.npy (default no)

usage() {
  cat <<EOF
Usage: $0 [--dataset_name NAME] [--start_sem K] [--start_spa K] [--resume] [--skip_dino] [--skip_dino_clean]

  --dataset_name NAME  Cambridge dataset folder name. Example: --dataset_name Cambridge_GreatCourt
                      (DINO path:  \$DINO_DIR/datasets/NAME/train/semantics)
                      (ACE  path:  datasets/NAME, output/NAME.pt)
  --start_sem K        Start from this k_sem value (must be in K_LIST). Example: --start_sem 32
  --start_spa K        Start from this k_spa value for the FIRST k_sem only (must be in K_LIST). Example: --start_spa 16
  --resume             Skip cells already filled (cell != "-"). Useful for continuing interrupted runs.

  --skip_dino          Do NOT run DINOv3 generation step. Assumes semantics already exist.
  --skip_dino_clean    When --skip_dino is set, still remove \$SEMANTICS_DIR/*.npy (normally not removed).

Env overrides:
  DATASET_NAME=Cambridge_GreatCourt $0
  START_SEM=32 START_SPA=16 RESUME=1 $0
  SKIP_DINO=1 $0
  TEST_TIMEOUT_SEC=600 MAX_TEST_RETRIES=2 RETRAIN_ON_TIMEOUT=1 $0 --resume
  CUDA_DEBUG_SYNC=1 $0   # helps surface device-side assert earlier
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_name)
      DATASET_NAME="$2"; shift 2;;
    --start_sem)
      START_SEM="$2"; shift 2;;
    --start_spa)
      START_SPA="$2"; shift 2;;
    --resume)
      RESUME=1; shift 1;;
    --skip_dino)
      SKIP_DINO=1; shift 1;;
    --skip_dino_clean)
      SKIP_DINO_CLEAN=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1;;
  esac
done

# =======================
# Derive dataset-dependent paths
# =======================
SEMANTICS_DIR="$DINO_DIR/datasets/${DATASET_NAME}/train/semantics"

ACE_DATASET="datasets/${DATASET_NAME}"
ACE_OUT="output/${DATASET_NAME}.pt"
ACE_TEST_FILE="$ACE_DIR/output/_test_result.txt"

# =======================
# Conda init (edit if needed)
# =======================
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Cannot find conda.sh. Please edit this script with the correct path."
  exit 1
fi

# =======================
# Helpers
# =======================
index_of_k() {
  local val="$1"
  local i
  for i in "${!K_LIST[@]}"; do
    if [[ "${K_LIST[$i]}" == "$val" ]]; then
      echo "$i"
      return 0
    fi
  done
  echo "-1"
}

parse_median_cm_deg() {
  python - <<'PY' "$1"
import re, sys
path = sys.argv[1]
txt = open(path, "r", encoding="utf-8", errors="ignore").read()
matches = re.findall(r"Median Error:\s*([0-9.]+)deg,\s*([0-9.]+)cm", txt)
if not matches:
    print("NA/NA")
    sys.exit(0)
deg, cm = matches[-1]
print(f"{float(cm):.2f}/{float(deg):.2f}")
PY
}

init_table() {
  local out="$1"
  : > "$out"
  printf "%-${CELL_W}s" "k_spa\\k_sem" >> "$out"
  for ksem in "${K_LIST[@]}"; do
    printf "%${CELL_W}d" "$ksem" >> "$out"
  done
  printf "\n" >> "$out"

  for kspa in "${K_LIST[@]}"; do
    printf "%-${CELL_W}d" "$kspa" >> "$out"
    for _ in "${K_LIST[@]}"; do
      printf "%${CELL_W}s" "-" >> "$out"
    done
    printf "\n" >> "$out"
  done
}

write_cell() {
  python - <<'PY' "$1" "$2" "$3" "$4" "$CELL_W"
import sys
table = sys.argv[1]
ri = int(sys.argv[2])
ci = int(sys.argv[3])
val = sys.argv[4]
W = int(sys.argv[5])

lines = open(table, "r", encoding="utf-8").read().splitlines()
target_row = ri + 1
line = lines[target_row]
parts = [line[i:i+W] for i in range(0, len(line), W)]
target_col = ci + 1
parts[target_col] = f"{val:>{W}}"
lines[target_row] = "".join(parts)
open(table, "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY
}

read_cell() {
  python - <<'PY' "$1" "$2" "$3" "$CELL_W"
import sys
table = sys.argv[1]
ri = int(sys.argv[2])
ci = int(sys.argv[3])
W = int(sys.argv[4])

lines = open(table, "r", encoding="utf-8").read().splitlines()
target_row = ri + 1
line = lines[target_row]
parts = [line[i:i+W] for i in range(0, len(line), W)]
target_col = ci + 1
print(parts[target_col].strip())
PY
}

run_with_timeout_killpg() {
  local timeout_sec="$1"; shift
  timeout --preserve-status --kill-after=10s --signal=TERM "$timeout_sec" \
    bash -lc 'set -e; exec "$@"' _ "$@"
}

kill_residuals_best_effort() {
  pkill -u "$USER" -f "python .*test.py" 2>/dev/null || true
  pkill -u "$USER" -f "python .*train.py" 2>/dev/null || true
}

env_prefix_for_test() {
  local prefix=""
  if [[ "$CUDA_DEBUG_SYNC" -eq 1 ]]; then
    prefix+="CUDA_LAUNCH_BLOCKING=1 "
  fi
  if [[ "$TORCH_STACKTRACE" -eq 1 ]]; then
    prefix+="TORCH_SHOW_CPP_STACKTRACES=1 "
  fi
  echo "$prefix"
}

# =======================
# Validate dataset paths
# =======================
echo "[Info] DATASET_NAME   = $DATASET_NAME"
echo "[Info] SEMANTICS_DIR  = $SEMANTICS_DIR"
echo "[Info] ACE_DATASET    = $ACE_DATASET"
echo "[Info] ACE_OUT        = $ACE_OUT"
echo "[Info] SKIP_DINO      = $SKIP_DINO (SKIP_DINO_CLEAN=$SKIP_DINO_CLEAN)"

# =======================
# Validate start positions
# =======================
SEM_START_IDX=0
SPA_START_IDX_DEFAULT=0

if [[ -n "${START_SEM}" ]]; then
  SEM_START_IDX="$(index_of_k "$START_SEM")"
  if [[ "$SEM_START_IDX" == "-1" ]]; then
    echo "[ERROR] --start_sem $START_SEM is not in K_LIST: ${K_LIST[*]}"
    exit 1
  fi
fi

if [[ -n "${START_SPA}" ]]; then
  SPA_START_IDX_DEFAULT="$(index_of_k "$START_SPA")"
  if [[ "$SPA_START_IDX_DEFAULT" == "-1" ]]; then
    echo "[ERROR] --start_spa $START_SPA is not in K_LIST: ${K_LIST[*]}"
    exit 1
  fi
fi

# =======================
# Main
# =======================
mkdir -p "$(dirname "$RESULTS_FILE")"

if [[ -f "$RESULTS_FILE" ]]; then
  echo "[Info] Found existing table -> $RESULTS_FILE"
else
  init_table "$RESULTS_FILE"
  echo "[Info] Initialized table -> $RESULTS_FILE"
fi

first_sem=1

for ((ci=SEM_START_IDX; ci<${#K_LIST[@]}; ci++)); do
  k_sem="${K_LIST[$ci]}"

  echo "============================================================"
  echo "[Grid] k_sem (feature clusters for semantics) = $k_sem"
  echo "============================================================"

  # =======================
  # DINOv3 (optional)
  # =======================
  if [[ "$SKIP_DINO" -eq 0 ]]; then
    echo "------------------------------------------------------------"
    echo "[DINOv3] Running semantic generation (k_sem=$k_sem)"
    echo "------------------------------------------------------------"

    conda activate "$ENV_DINO"
    cd "$DINO_DIR"

    python "$DINO_SCRIPT" --K "$k_sem"

    echo "[DINOv3] Removing *.npy in $SEMANTICS_DIR"
    rm -f "$SEMANTICS_DIR"/*.npy || true
  else
    echo "------------------------------------------------------------"
    echo "[DINOv3] Skipped (SKIP_DINO=1). Using existing semantics at:"
    echo "         $SEMANTICS_DIR"
    echo "------------------------------------------------------------"
    if [[ "$SKIP_DINO_CLEAN" -eq 1 ]]; then
      echo "[DINOv3] (skip mode) Removing *.npy in $SEMANTICS_DIR"
      rm -f "$SEMANTICS_DIR"/*.npy || true
    fi
  fi

  spa_start_idx=0
  if [[ "$first_sem" == "1" ]]; then
    spa_start_idx="$SPA_START_IDX_DEFAULT"
  fi

  for ((ri=spa_start_idx; ri<${#K_LIST[@]}; ri++)); do
    k_spa="${K_LIST[$ri]}"

    if [[ "$RESUME" == "1" ]]; then
      cur_cell="$(read_cell "$RESULTS_FILE" "$ri" "$ci")"
      if [[ "$cur_cell" != "-" && "$cur_cell" != "" ]]; then
        echo "[Skip] Already filled (k_spa=$k_spa, k_sem=$k_sem) -> $cur_cell"
        continue
      fi
    fi

    echo "-----------------------------"
    echo "[ACE] k_spa (spatial clusters) = $k_spa ; k_sem = $k_sem"
    echo "-----------------------------"

    conda activate "$ENV_ACE"
    cd "$ACE_DIR"

    rm -f "$ACE_TEST_FILE" || true

    echo "[ACE][Train] Training..."
    python train.py "$ACE_DATASET" "$ACE_OUT" \
      --feature_clusters "$k_sem" \
      --spatial_clusters "$k_spa"

    echo "[ACE][Test] Testing with timeout watchdog..."
    test_ok=0
    attempt=0

    while [[ "$attempt" -le "$MAX_TEST_RETRIES" ]]; do
      attempt=$((attempt+1))
      echo "[ACE][Test] Attempt $attempt / $((MAX_TEST_RETRIES+1)) (timeout=${TEST_TIMEOUT_SEC}s)"

      prefix="$(env_prefix_for_test)"

      set +e
      run_with_timeout_killpg "$TEST_TIMEOUT_SEC" \
        bash -lc "${prefix}python test.py \"$ACE_DATASET\" \"$ACE_OUT\" --feature_clusters \"$k_sem\" --spatial_clusters \"$k_spa\""
      rc=$?
      set -e

      if [[ "$rc" -eq 0 ]]; then
        test_ok=1
        break
      fi

      if [[ "$rc" -eq 124 ]]; then
        echo "[ACE][Test][TIMEOUT] test.py exceeded ${TEST_TIMEOUT_SEC}s. rc=$rc"
      else
        echo "[ACE][Test][FAIL] test.py exited with rc=$rc"
      fi

      kill_residuals_best_effort

      if [[ "$RETRAIN_ON_TIMEOUT" -eq 1 && "$attempt" -le "$MAX_TEST_RETRIES" ]]; then
        echo "[ACE] Re-training due to test failure/timeout..."
        rm -f "$ACE_TEST_FILE" || true

        python train.py "$ACE_DATASET" "$ACE_OUT" \
          --feature_clusters "$k_sem" \
          --spatial_clusters "$k_spa"

        echo "[ACE] Re-test after retraining..."
      else
        break
      fi
    done

    if [[ "$ACE_TEST_FILE" == "" ]]; then
      echo "[Warn] ACE_TEST_FILE is empty string. Check path settings."
    fi

    if [[ -f "$ACE_TEST_FILE" && "$test_ok" -eq 1 ]]; then
      cell="$(parse_median_cm_deg "$ACE_TEST_FILE")"
    else
      cell="NA/NA"
      echo "[Warn] Missing/invalid $ACE_TEST_FILE (or test failed)."
    fi

    write_cell "$RESULTS_FILE" "$ri" "$ci" "$cell"
    echo "[Saved] (k_spa=$k_spa, k_sem=$k_sem) -> $cell"
  done

  first_sem=0
done

echo "============================================================"
echo "[DONE] Results saved to:"
echo "  $RESULTS_FILE"
echo "============================================================"