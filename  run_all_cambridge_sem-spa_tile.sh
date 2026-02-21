#!/usr/bin/env bash
set -euo pipefail

# =======================
# Config
# =======================
RUN_SCRIPT="${RUN_SCRIPT:-./run_sem-spa_statistics-tile.sh}"

DATASETS=(                         # cols
  "Cambridge_GreatCourt"
  "Cambridge_KingsCollege"
  "Cambridge_OldHospital"
  "Cambridge_ShopFacade"
  "Cambridge_StMarysChurch"
)

# 把 driver 接收到的参数原样传给子脚本
# 例如：--resume / --start_sem 32 / --start_spa 16
PASSTHROUGH_ARGS=("$@")

# =======================
# Checks
# =======================
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "[ERROR] RUN_SCRIPT not found: $RUN_SCRIPT"
  echo "        You can set it like:"
  echo "        RUN_SCRIPT=/path/to/run_sem-spa_statistics-tile.sh $0"
  exit 1
fi

chmod +x "$RUN_SCRIPT" 2>/dev/null || true

# =======================
# Run
# =======================
for ds in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "[Driver] Running dataset: $ds"
  echo "[Driver] Results file:"
  echo "  \$ACE_DIR/output/semantic-spatial_clustering_results_${ds}.txt"
  echo "============================================================"

  bash "$RUN_SCRIPT" \
    --dataset_name "$ds" \
    "${PASSTHROUGH_ARGS[@]}"
done

echo "============================================================"
echo "[Driver] DONE. All Cambridge datasets finished."
echo "============================================================"