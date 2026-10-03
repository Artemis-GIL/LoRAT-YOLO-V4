#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/xu/miniconda3/envs/mvlm_cvpr26_cuda128/bin/python
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
V3_OUT=/home/xu/zyx/send/LORAT_YOLO_TEST420_V3_20261003
ARGS=()
while (($#)); do
  case "$1" in
    --output) V3_OUT="$2"; shift 2 ;;
    --dry-run|--evaluate-only) ARGS+=("$1"); shift ;;
    --dataset-root|--frames-root|--sequence|--max-frames|--config)
      ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help)
      "$PYTHON" "$SCRIPT_DIR/test420_v3.py" --help
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
mkdir -p "$V3_OUT"
exec > >(tee -a "$V3_OUT/console.log") 2>&1
printf 'V3 Test420 output: %s\n' "$V3_OUT"
"$PYTHON" -u "$SCRIPT_DIR/test420_v3.py" --output "$V3_OUT" "${ARGS[@]}"
