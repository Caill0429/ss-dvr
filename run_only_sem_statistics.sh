#!/usr/bin/env bash
set -eo pipefail

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

# =======================
# Grid settings (NEW)
# =======================
K_SPA=1
K_SEM_LIST=(1 2 4 8 16 32 64)

# Cambridge subsets -> columns
DATASETS=(
  "Cambridge_GreatCourt"
  "Cambridge_KingsCollege"
  "Cambridge_OldHospital"
  "Cambridge_ShopFacade"
  "Cambridge_StMarysChurch"
)

# Conda env names
ENV_DINO="dinov3"
ENV_ACE="ace"

# Paths
DINO_DIR="$HOME/PycharmProjects/DINOv3/dinov3"
ACE_DIR="$HOME/PycharmProjects/ss-dvr/ss-dvr"

# DINOv3 script
DINO_SCRIPT="dinov3_semantic_label_generation-2.py"

# Table format
CELL_W=18

# =======================
# Timeout & retry policy
# =======================
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-900}"
RETRAIN_ON_TIMEOUT="${RETRAIN_ON_TIMEOUT:-1}"
MAX_TEST_RETRIES="${MAX_TEST_RETRIES:-1}"
CUDA_DEBUG_SYNC="${CUDA_DEBUG_SYNC:-0}"
TORCH_STACKTRACE="${TORCH_STACKTRACE:-1}"

# =======================
# CLI options
# =======================
START_SEM="${START_SEM:-}"         # value in K_SEM_LIST, e.g. 16
START_DATASET="${START_DATASET:-}" # dataset name, e.g. Cambridge_OldHospital
RESUME=0

# DINO control (NEW)
SKIP_DINO=1            # default: skip DINO
REQUIRE_SEMANTICS=0    # default: do not hard-fail if semantics missing (ACE may fail anyway)

usage() {
  cat <<EOF
Usage: $0 [--start_sem K] [--start_dataset NAME] [--resume]
          [--run_dino | --skip_dino] [--require_semantics]

Grid:
  K_SPA fixed = 1
  K_SEM_LIST   = ${K_SEM_LIST[*]}
  DATASETS     = ${DATASETS[*]}

Options:
  --start_sem K           Start from this k_sem value (must be in K_SEM_LIST).
  --start_dataset NAME    Start from this dataset column. Example: Cambridge_OldHospital
  --resume                Skip cells already filled (cell != "-").

DINOv3:
  --skip_dino             Do NOT run DINOv3 (default).
  --run_dino              Run DINOv3 each cell to generate semantics.
  --require_semantics     If semantics dir missing/empty, exit with error (useful with --skip_dino).

Env overrides:
  START_SEM=16 START_DATASET=Cambridge_OldHospital RESUME=1 $0
  TEST_TIMEOUT_SEC=600 MAX_TEST_RETRIES=2 RETRAIN_ON_TIMEOUT=1 $0 --resume
  CUDA_DEBUG_SYNC=1 $0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start_sem)
      START_SEM="$2"; shift 2;;
    --start_dataset)
      START_DATASET="$2"; shift 2;;
    --resume)
      RESUME=1; shift 1;;
    --skip_dino)
      SKIP_DINO=1; shift 1;;
    --run_dino)
      SKIP_DINO=0; shift 1;;
    --require_semantics)
      REQUIRE_SEMANTICS=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1;;
  esac
done

# =======================
# Conda init
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
index_of_item() {
  local target="$1"; shift
  local arr=("$@")
  local i
  for i in "${!arr[@]}"; do
    if [[ "${arr[$i]}" == "$target" ]]; then
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

# ---- table helpers (global table: rows=k_sem, cols=dataset) ----
RESULTS_FILE="$ACE_DIR/output/semantic-spatial_clustering_results_KSPA1.txt"

init_global_table() {
  local out="$1"
  : > "$out"
  printf "%-${CELL_W}s" "k_sem\\dataset" >> "$out"
  for ds in "${DATASETS[@]}"; do
    printf "%${CELL_W}s" "$ds" >> "$out"
  done
  printf "\n" >> "$out"

  for ksem in "${K_SEM_LIST[@]}"; do
    printf "%-${CELL_W}d" "$ksem" >> "$out"
    for _ in "${DATASETS[@]}"; do
      printf "%${CELL_W}s" "-" >> "$out"
    done
    printf "\n" >> "$out"
  done
}

write_cell_global() {
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

read_cell_global() {
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

semantics_nonempty() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  # accept either png/npz/pt/npy depending on your pipeline; extend if needed
  ls "$dir"/*.png "$dir"/*.npz "$dir"/*.pt "$dir"/*.npy >/dev/null 2>&1
}

# =======================
# Validate start indices
# =======================
SEM_START_IDX=0
DS_START_IDX=0

if [[ -n "${START_SEM}" ]]; then
  SEM_START_IDX="$(index_of_item "$START_SEM" "${K_SEM_LIST[@]}")"
  if [[ "$SEM_START_IDX" == "-1" ]]; then
    echo "[ERROR] --start_sem $START_SEM is not in K_SEM_LIST: ${K_SEM_LIST[*]}"
    exit 1
  fi
fi

if [[ -n "${START_DATASET}" ]]; then
  DS_START_IDX="$(index_of_item "$START_DATASET" "${DATASETS[@]}")"
  if [[ "$DS_START_IDX" == "-1" ]]; then
    echo "[ERROR] --start_dataset $START_DATASET is not in DATASETS: ${DATASETS[*]}"
    exit 1
  fi
fi

# =======================
# Init results table
# =======================
mkdir -p "$(dirname "$RESULTS_FILE")"
if [[ -f "$RESULTS_FILE" ]]; then
  echo "[Info] Found existing table -> $RESULTS_FILE"
else
  init_global_table "$RESULTS_FILE"
  echo "[Info] Initialized table -> $RESULTS_FILE"
fi

echo "[Info] K_SPA fixed         = $K_SPA"
echo "[Info] SKIP_DINO           = $SKIP_DINO (0=run,1=skip)"
echo "[Info] REQUIRE_SEMANTICS   = $REQUIRE_SEMANTICS"
echo "[Info] RESULTS_FILE        = $RESULTS_FILE"

# =======================
# Main loops
# rows: k_sem, cols: dataset
# =======================
for ((ri=SEM_START_IDX; ri<${#K_SEM_LIST[@]}; ri++)); do
  k_sem="${K_SEM_LIST[$ri]}"

  echo "============================================================"
  echo "[Loop] k_sem = $k_sem  (K_SPA fixed = $K_SPA)"
  echo "============================================================"

  ds_start_idx=0
  if [[ "$ri" == "$SEM_START_IDX" ]]; then
    ds_start_idx="$DS_START_IDX"
  fi

  for ((ci=ds_start_idx; ci<${#DATASETS[@]}; ci++)); do
    DATASET_NAME="${DATASETS[$ci]}"

    if [[ "$RESUME" == "1" ]]; then
      cur_cell="$(read_cell_global "$RESULTS_FILE" "$ri" "$ci")"
      if [[ "$cur_cell" != "-" && "$cur_cell" != "" ]]; then
        echo "[Skip] Already filled (k_sem=$k_sem, dataset=$DATASET_NAME) -> $cur_cell"
        continue
      fi
    fi

    # derive dataset paths
    SEMANTICS_DIR="$DINO_DIR/datasets/${DATASET_NAME}/train/semantics"
    ACE_DATASET="datasets/${DATASET_NAME}"
    ACE_OUT="output/${DATASET_NAME}.pt"
    ACE_TEST_FILE="$ACE_DIR/output/_test_result.txt"

    echo "-----------------------------"
    echo "[DATASET] $DATASET_NAME"
    echo "[SEM] k_sem=$k_sem ; [SPA] k_spa=$K_SPA"
    echo "[Paths] SEMANTICS_DIR=$SEMANTICS_DIR"
    echo "[Paths] ACE_DATASET=$ACE_DATASET"
    echo "[Paths] ACE_OUT=$ACE_OUT"
    echo "-----------------------------"

    # 1) DINO semantics (OPTIONAL)
    if [[ "$SKIP_DINO" -eq 1 ]]; then
      echo "[DINOv3] SKIPPED (use --run_dino to enable)."
      if [[ "$REQUIRE_SEMANTICS" -eq 1 ]]; then
        if ! semantics_nonempty "$SEMANTICS_DIR"; then
          echo "[ERROR] --require_semantics set, but semantics missing/empty:"
          echo "        $SEMANTICS_DIR"
          exit 1
        fi
      fi
    else
      conda activate "$ENV_DINO"
      cd "$DINO_DIR"

      # check whether DINO_SCRIPT supports --dataset
      set +e
      help_txt="$(python "$DINO_SCRIPT" --help 2>&1)"
      rc=$?
      set -e
      if [[ "$rc" -ne 0 ]]; then
        echo "[ERROR] Failed to run: python $DINO_SCRIPT --help"
        echo "$help_txt"
        exit 1
      fi

      if echo "$help_txt" | grep -q -- "--dataset"; then
        echo "[DINOv3] Running: python $DINO_SCRIPT --dataset $DATASET_NAME --K $k_sem"
        python "$DINO_SCRIPT" --dataset "$DATASET_NAME" --K "$k_sem"
      else
        echo "[DINOv3] Running: python $DINO_SCRIPT --K $k_sem  (no --dataset support detected)"
        python "$DINO_SCRIPT" --K "$k_sem"
      fi

      # keep your original cleanup behavior
      echo "[DINOv3] Removing *.npy in $SEMANTICS_DIR"
      rm -f "$SEMANTICS_DIR"/*.npy || true

      if [[ "$REQUIRE_SEMANTICS" -eq 1 ]]; then
        if ! semantics_nonempty "$SEMANTICS_DIR"; then
          echo "[ERROR] DINO finished but semantics still missing/empty:"
          echo "        $SEMANTICS_DIR"
          exit 1
        fi
      fi
    fi

    # 2) ACE train + test
    conda activate "$ENV_ACE"
    cd "$ACE_DIR"

    rm -f "$ACE_TEST_FILE" || true

    echo "[ACE][Train] Training..."
    python train.py "$ACE_DATASET" "$ACE_OUT" \
      --feature_clusters "$k_sem" \
      --spatial_clusters "$K_SPA"

    echo "[ACE][Test] Testing with timeout watchdog..."
    test_ok=0
    attempt=0

    while [[ "$attempt" -le "$MAX_TEST_RETRIES" ]]; do
      attempt=$((attempt+1))
      echo "[ACE][Test] Attempt $attempt / $((MAX_TEST_RETRIES+1)) (timeout=${TEST_TIMEOUT_SEC}s)"

      prefix="$(env_prefix_for_test)"

      set +e
      run_with_timeout_killpg "$TEST_TIMEOUT_SEC" \
        bash -lc "${prefix}python test.py \"$ACE_DATASET\" \"$ACE_OUT\" --feature_clusters \"$k_sem\" --spatial_clusters \"$K_SPA\""
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
          --spatial_clusters "$K_SPA"
        echo "[ACE] Re-test after retraining..."
      else
        break
      fi
    done

    if [[ -f "$ACE_TEST_FILE" && "$test_ok" -eq 1 ]]; then
      cell="$(parse_median_cm_deg "$ACE_TEST_FILE")"
    else
      cell="NA/NA"
      echo "[Warn] Missing/invalid $ACE_TEST_FILE (or test failed)."
    fi

    write_cell_global "$RESULTS_FILE" "$ri" "$ci" "$cell"
    echo "[Saved] (k_sem=$k_sem, dataset=$DATASET_NAME, k_spa=$K_SPA) -> $cell"
  done
done

echo "============================================================"
echo "[DONE] Results saved to:"
echo "  $RESULTS_FILE"
echo "  (K_SPA fixed = $K_SPA)"
echo "============================================================"