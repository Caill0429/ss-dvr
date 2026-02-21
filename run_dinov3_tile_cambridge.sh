#!/usr/bin/env bash
set -eo pipefail

# =======================
# Defaults (can override)
# =======================
K_SEM="${K_SEM:-16}"   # default k_sem
K_SPA="${K_SPA:-4}"    # default k_spa

# Conda env names
ENV_DINO="dinov3"
ENV_ACE="ace"

# Paths
DINO_DIR="$HOME/lanyun-tmp/dinov3"
ACE_DIR="$HOME/lanyun-tmp/ace"

DINO_SCRIPT="dinov3_semantic_label_generation-2.py"

# Results (one line per dataset)
RESULTS_FILE="$ACE_DIR/output/cambridge_ksem${K_SEM}_kspa${K_SPA}.txt"

# Whether to run DINO step (1=run, 0=skip)
RUN_DINO="${RUN_DINO:-1}"

# test timeout (seconds)
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-900}"

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

# =======================
# Cambridge dataset list
# (match your folder names)
# =======================
DATASETS=(
  "Cambridge_GreatCourt"
  "Cambridge_KingsCollege"
  "Cambridge_OldHospital"
  "Cambridge_ShopFacade"
  "Cambridge_StMarysChurch"
)

# =======================
# Main
# =======================
mkdir -p "$(dirname "$RESULTS_FILE")"
: > "$RESULTS_FILE"
echo -e "dataset\tresult(cm/deg)\tinfo" >> "$RESULTS_FILE"

echo "============================================================"
echo "[Config] k_sem=${K_SEM}, k_spa=${K_SPA}, RUN_DINO=${RUN_DINO}"
echo "[Output] $RESULTS_FILE"
echo "============================================================"

for DS in "${DATASETS[@]}"; do
  echo ""
  echo "============================================================"
  echo "[DATASET] $DS"
  echo "============================================================"

  # Paths per dataset
  ACE_DATASET="datasets/${DS}"
  ACE_OUT="output/${DS}.pt"
  ACE_TEST_FILE="$ACE_DIR/output/_test_result.txt"

  # DINO semantics folder (if you need it for cleanup or checking)
  SEMANTICS_DIR="$DINO_DIR/datasets/${DS}/train/semantics"

  # -----------------------
  # 1) DINOv3 semantic label generation (optional)
  # -----------------------
  if [[ "$RUN_DINO" -eq 1 ]]; then
    echo "[DINOv3] Generating semantic labels: DS=$DS, K=$K_SEM"
    conda activate "$ENV_DINO"
    cd "$DINO_DIR"

    # NOTE:
    # - 如果你的脚本不支持 --dataset，请把下面这一行改成你脚本实际的参数形式
    # - 如果脚本内部写死了 Cambridge_OldHospital，那么就需要你在脚本里加一个 dataset 参数
    python "$DINO_SCRIPT" --K "$K_SEM" --dataset "$DS"

    echo "[DINOv3] Removing *.npy in $SEMANTICS_DIR"
    rm -f "$SEMANTICS_DIR"/*.npy 2>/dev/null || true
  else
    echo "[DINOv3] Skipped."
  fi

  # -----------------------
  # 2) ACE train
  # -----------------------
  echo "[ACE][Train] Training: $ACE_DATASET -> $ACE_OUT"
  conda activate "$ENV_ACE"
  cd "$ACE_DIR"

  rm -f "$ACE_TEST_FILE" 2>/dev/null || true

  python train.py "$ACE_DATASET" "$ACE_OUT" \
    --feature_clusters "$K_SEM" \
    --spatial_clusters "$K_SPA"

  # -----------------------
  # 3) ACE test with timeout
  # -----------------------
  echo "[ACE][Test] Testing with timeout=${TEST_TIMEOUT_SEC}s"
  set +e
  run_with_timeout_killpg "$TEST_TIMEOUT_SEC" \
    python test.py "$ACE_DATASET" "$ACE_OUT" \
      --feature_clusters "$K_SEM" \
      --spatial_clusters "$K_SPA"
  rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    echo "[ACE][Test][FAIL] rc=$rc"
    echo -e "${DS}\tNA/NA\ttest_rc=${rc}" >> "$RESULTS_FILE"
    continue
  fi

  # parse result
  if [[ -f "$ACE_TEST_FILE" ]]; then
    cell="$(parse_median_cm_deg "$ACE_TEST_FILE")"
    echo "[RESULT] $DS -> $cell"
    echo -e "${DS}\t${cell}\tok" >> "$RESULTS_FILE"
  else
    echo "[WARN] Missing $ACE_TEST_FILE"
    echo -e "${DS}\tNA/NA\tmissing_test_file" >> "$RESULTS_FILE"
  fi
done

echo ""
echo "============================================================"
echo "[DONE] Results saved to:"
echo "  $RESULTS_FILE"
echo "============================================================"