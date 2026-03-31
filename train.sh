#!/bin/bash

# 训练脚本
# 使用方法: bash train.sh

python train.py \
    --data_dir "pcot_persu_data" \
    --train_files "gpt-5-mini-2025-08-07_ECTF_train.csv","gpt-5-mini-2025-08-07_CoAID_train.csv" \
    --val_files "gpt-5-mini-2025-08-07_ECTF_validation.csv","gpt-5-mini-2025-08-07_CoAID_validation.csv" \
    --test_files "gpt-5-mini-2025-08-07_ECTF_test.csv","gpt-5-mini-2025-08-07_CoAID_test.csv","gpt-5-mini-2025-08-07_EUDisinfo_test.csv","gpt-5-mini-2025-08-07_ISOTFakeNews_test.csv","gpt-5-mini-2025-08-07_MultiDis_test.csv" \
    --save_path "models/persuasion_mlp.pth"