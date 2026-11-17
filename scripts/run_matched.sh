#!/usr/bin/env bash
# SSD weighted finals, then YOLOS CV and finals on the SSD/RetinaNet weighted setup, its pitch32-vs-v2 comparison,
# and one results file for every setup. Each step is safe to rerun after a stop: finished work is skipped.
cd "$(dirname "$0")/.." || exit 1
YOLO="${YOLO:-$HOME/miniconda3/envs/mtp_yolos/bin/python}"
MATCHED="yolos_pitch32_weighted_matched yolos_v2_weighted_matched"
"$YOLO" scripts/final_test.py --setups ssd_pitch32_weighted ssd_v2_weighted 2>&1 | tee -a final_test.log
"$YOLO" scripts/cv5.py --setups $MATCHED 2>&1 | tee -a cv5.log
"$YOLO" scripts/final_test.py --setups $MATCHED 2>&1 | tee -a final_test.log
"$YOLO" scripts/compare_cv.py yolos_pitch32_weighted_matched yolos_v2_weighted_matched 2>&1 | tee -a compare.log
"$YOLO" scripts/collect_results.py 2>&1 | tee -a collect.log
echo "RUN_MATCHED ENDED"
