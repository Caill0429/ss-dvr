#!/usr/bin/env bash
set -eo pipefail

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

# =======================
# Settings
# =======================
K_SEM="${K_SEM:-64}"               # adaptive semantic labels cap -> proto_max
K_SPA_LIST=(1)      # rows
DATASETS=(                         # cols
  "Cambridge_GreatCourt"
  "Cambridge_KingsCollege"
  "Cambridge_OldHospital"
  "Cambridge_ShopFacade"
  "Cambridge_StMarysChurch"
)

ENV_DINO="dinov3"
ENV_ACE="ace"

DINO_DIR="$HOME/PycharmProjects/DINOv3/dinov3"
ACE_DIR="$HOME/PycharmProjects/ss-dvr/ss-dvr"

DINO_SCRIPT="dinov3_semantic_label_generation_adapter.py"

TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-900}"
RUN_DINO="${RUN_DINO:-1}"
RESUME="${RESUME:-0}"
IMG_NUM="${IMG_NUM:-0}"

RESULTS_FILE="$ACE_DIR/output/cambridge_grid_adaptSem$K_SEM.txt"
CELL_W=18

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

init_table() {
  local out="$1"
  : > "$out"

  # header row
  printf "%-${CELL_W}s" "k_spa\\ds" >> "$out"
  for ds in "${DATASETS[@]}"; do
    printf "%${CELL_W}s" "$ds" >> "$out"
  done
  printf "\n" >> "$out"

  # data rows
  for kspa in "${K_SPA_LIST[@]}"; do
    printf "%-${CELL_W}s" "$kspa" >> "$out"
    for _ in "${DATASETS[@]}"; do
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
target_row = ri + 1  # skip header
line = lines[target_row]
parts = [line[i:i+W] for i in range(0, len(line), W)]
target_col = ci + 1  # skip row header
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

echo "============================================================"
echo "[Config] by-dataset (column-wise) running"
echo "  adapt_sem_max(K_SEM/proto_max)=$K_SEM"
echo "  k_spa rows=(${K_SPA_LIST[*]})"
echo "  datasets cols=(${DATASETS[*]})"
echo "[Output] $RESULTS_FILE"
echo "============================================================"

# Outer loop: dataset (columns)
for ((ci=0; ci<${#DATASETS[@]}; ci++)); do
  DS="${DATASETS[$ci]}"
  echo ""
  echo "============================================================"
  echo "[DATASET] $DS (col=$ci)"
  echo "============================================================"

if [[ "$RUN_DINO" -eq 1 ]]; then
  echo "[DINOv3] Generating adaptive semantics ONCE: dataset=$DS, proto_max=$K_SEM"
  conda activate "$ENV_DINO"
  cd "$DINO_DIR"
  python "$DINO_SCRIPT" --dataset "$DS" --proto_max "$K_SEM" --proto_merge_every "$IMG_NUM"


  SEMANTICS_DIR="$DINO_DIR/datasets/${DS}/train/semantics"
  echo "[DINOv3] Removing *.npy in $SEMANTICS_DIR"
  rm -f "$SEMANTICS_DIR"/*.npy 2>/dev/null || true
else
  echo "[DINOv3] Skipped (RUN_DINO=0)."
fi

  # ACE shared paths per dataset
  ACE_DATASET="datasets/${DS}"
  ACE_TEST_FILE="$ACE_DIR/output/_test_result.txt"

  # Inner loop: k_spa (rows)
  for ((ri=0; ri<${#K_SPA_LIST[@]}; ri++)); do
    k_spa="${K_SPA_LIST[$ri]}"

    if [[ "$RESUME" == "1" ]]; then
      cur="$(read_cell "$RESULTS_FILE" "$ri" "$ci")"
      if [[ "$cur" != "-" && "$cur" != "" ]]; then
        echo "[Skip] Already filled (k_spa=$k_spa, dataset=$DS) -> $cur"
        continue
      fi
    fi

    echo "------------------------------"
    echo "[ACE] dataset=$DS | k_spa=$k_spa | k_sem(adapt cap)=$K_SEM"
    echo "------------------------------"

    ACE_OUT="output/${DS}.pt"

    conda activate "$ENV_ACE"
    cd "$ACE_DIR"

    rm -f "$ACE_TEST_FILE" 2>/dev/null || true

    echo "[ACE][Train] ..."
    python train.py "$ACE_DATASET" "$ACE_OUT" \
      --feature_clusters "$K_SEM" \
      --spatial_clusters "$k_spa"

    echo "[ACE][Test] timeout=${TEST_TIMEOUT_SEC}s"
    set +e
    run_with_timeout_killpg "$TEST_TIMEOUT_SEC" \
      python test.py "$ACE_DATASET" "$ACE_OUT" \
        --feature_clusters "$K_SEM" \
        --spatial_clusters "$k_spa"
    rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
      echo "[ACE][Test][FAIL] rc=$rc"
      cell="NA/NA"
    else
      if [[ -f "$ACE_TEST_FILE" ]]; then
        cell="$(parse_median_cm_deg "$ACE_TEST_FILE")"
      else
        echo "[WARN] Missing $ACE_TEST_FILE"
        cell="NA/NA"
      fi
    fi

    write_cell "$RESULTS_FILE" "$ri" "$ci" "$cell"
    echo "[Saved] (dataset=$DS, k_spa=$k_spa) -> $cell"
  done
done

echo "============================================================"
echo "[DONE] Results saved to:"
echo "  $RESULTS_FILE"
echo "============================================================"