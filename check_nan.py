#!/usr/bin/env python
"""检查所有生成数据中generated_pred为nan的比例"""

import os
import pandas as pd
import glob

data_dir = "pcot_persu_data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

print(f"检查 {len(csv_files)} 个CSV文件中的 generated_pred 列\n")
print(f"{'文件':<60} {'总数':>8} {'NaN数':>8} {'比例':>10}")
print("-" * 90)

total_samples = 0
total_nan = 0

for csv_file in sorted(csv_files):
    filename = os.path.basename(csv_file)
    df = pd.read_csv(csv_file)

    total = len(df)
    nan_count = df['generated_pred'].isna().sum()

    total_samples += total
    total_nan += nan_count

    ratio = (nan_count / total * 100) if total > 0 else 0
    print(f"{filename:<60} {total:>8} {nan_count:>8} {ratio:>9.2f}%")

print("-" * 90)
overall_ratio = (total_nan / total_samples * 100) if total_samples > 0 else 0
print(f"{'总计':<60} {total_samples:>8} {total_nan:>8} {overall_ratio:>9.2f}%")