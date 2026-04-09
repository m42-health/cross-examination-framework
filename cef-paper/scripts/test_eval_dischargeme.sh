#!/bin/bash
# Test script for DischargeME evaluation

echo "Testing DischargeME Evaluation Pipeline"
echo "========================================"
echo ""

# Set paths
PREDICTIONS_PATH="/data/cef/dischargeme/dischargeme/DeepSeek-V3.1/predictions/DeepSeek-V3.1_raw_bhc_responses_tmp.jsonl"
REFERENCE_PATH="/data/cef/dischargeme/gt/discharge-me/1.3"
RESULTS_PATH="/data/cef/dischargeme/results_cef"

# Change to evaluation directory
cd /home/yathagata/cef-translation/src/evaluation

echo "Test 1: Traditional Metrics Only (Debug Mode)"
echo "----------------------------------------------"
python eval_dischargeme.py \
  --predictions_path="${PREDICTIONS_PATH}" \
  --reference_path="${REFERENCE_PATH}" \
  --results_base_path="${RESULTS_PATH}" \
  --eval_traditional=True \
  --eval_cef=False \
  --debug=True

echo ""
echo "Test completed! Check results in ${RESULTS_PATH}/debug"
echo ""
echo "To run CEF evaluation, use:"
echo "python eval_dischargeme.py \\"
echo "  --predictions_path=\"${PREDICTIONS_PATH}\" \\"
echo "  --worker=worker-7 \\"
echo "  --judge_worker=worker-12 \\"
echo "  --eval_cef=True \\"
echo "  --eval_traditional=True \\"
echo "  --num_questions_to_generate=10 \\"
echo "  --debug=True"

python eval_dischargeme.py \
  --predictions_path="/data/cef/dischargeme/dischargeme/DeepSeek-V3.1/predictions/DeepSeek-V3.1_raw_bhc_responses_tmp.jsonl" \
  --worker=worker-10 \
  --judge_worker=worker-13 \
  --eval_cef=True \
  --num_questions_to_generate=10 \
  --debug=True