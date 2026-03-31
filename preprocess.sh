#!/bin/bash

# 预处理数据集脚本
# 使用方法: bash preprocess.sh

model="gpt-5-mini-2025-08-07"
output_dir="pcot_persu_data"

# 定义数据集列表 (格式: "原始数据路径 输出文件名")
datasets=(
    # "raw_data/ECTF/train.csv ${model}_ECTF_train.csv"
    # "raw_data/ECTF/validation.csv ${model}_ECTF_validation.csv"
    # "raw_data/ECTF/test.csv ${model}_ECTF_test.csv"
    # "raw_data/CoAID/train.csv ${model}_CoAID_train.csv"
    # "raw_data/CoAID/validation.csv ${model}_CoAID_validation.csv"
    # "raw_data/CoAID/test.csv ${model}_CoAID_test.csv"
    # "raw_data/ISOTFakeNews/test.csv ${model}_ISOTFakeNews_test.csv"
    "raw_data/MultiDis/test.csv ${model}_MultiDis_test.csv"
    "raw_data/EUDisinfo/test.csv ${model}_EUDisinfo_test.csv"
)

# 遍历处理
for item in "${datasets[@]}"; do
    IFS=' ' read -r input_file output_file <<< "$item"

    echo "=========================================="
    echo "Processing: $input_file -> $output_dir/$output_file"
    echo "=========================================="

    python data/preprocess.py \
        --input "$input_file" \
        --output "$output_dir/$output_file" \
        --model "$model"
done

echo "All preprocessing completed!"