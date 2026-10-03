#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/xu/miniconda3/envs/mvlm_cvpr26_cuda128/bin/python
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
V4_OUT=/home/xu/zyx/send/LORAT_YOLO_TEST420_V4_20261003
ARGS=()
while (($#)); do
  case "$1" in
    --output) V4_OUT="$2"; shift 2 ;;
    --dry-run|--evaluate-only) ARGS+=("$1"); shift ;;
    --dataset-root|--frames-root|--sequence|--max-frames|--config)
      ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help)
      "$PYTHON" "$SCRIPT_DIR/test420_v4.py" --help
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
mkdir -p "$V4_OUT"
exec > >(tee -a "$V4_OUT/console.log") 2>&1
printf 'V4 Test420 output: %s\n' "$V4_OUT"
"$PYTHON" -u "$SCRIPT_DIR/test420_v4.py" --output "$V4_OUT" "${ARGS[@]}"
