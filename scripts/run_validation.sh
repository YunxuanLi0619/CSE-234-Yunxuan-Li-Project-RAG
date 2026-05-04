#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
OUT="${1:-$ROOT/logs/validation_output.json}"
REPORT="${2:-$ROOT/logs/retrieval_report.json}"

python3 "$ROOT/main.py" \
  --input "$PROJECT_ROOT/validation-set-golden-qa-pairs.json" \
  --output "$OUT" \
  --quiet

python3 "$PROJECT_ROOT/Metrics/evaluate_retrieval.py" \
  --output "$OUT" \
  --validation "$PROJECT_ROOT/validation-set-golden-qa-pairs.json" \
  --k 5 > "$REPORT"

echo "Wrote output: $OUT"
echo "Wrote retrieval report: $REPORT"

