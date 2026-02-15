#!/bin/bash
# ============================================
# Retrain Script — Credit Card Fraud Detection
# ============================================
# Usage:
#   ./retrain.sh
#
# Cron example (weekly at 3 AM Sunday):
#   0 3 * * 0 /path/to/mlops-credit-fraud/retrain.sh
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/retrain.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

mkdir -p "$LOG_DIR"

echo "============================================" | tee -a "$LOG_FILE"
echo "[$TIMESTAMP] Starting retrain pipeline..." | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Check dataset exists
if [ ! -f "data/creditcard.csv" ]; then
    echo "[$TIMESTAMP] ERROR: data/creditcard.csv not found!" | tee -a "$LOG_FILE"
    echo "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" | tee -a "$LOG_FILE"
    exit 1
fi

# Run the pipeline
echo "[$TIMESTAMP] Running model_pipeline.py..." | tee -a "$LOG_FILE"
python3 model_pipeline.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] ✅ Retrain completed successfully." | tee -a "$LOG_FILE"
else
    echo "[$TIMESTAMP] ❌ Retrain failed with exit code $EXIT_CODE." | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "" >> "$LOG_FILE"
