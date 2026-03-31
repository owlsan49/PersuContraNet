"""
预处理脚本 - 使用大模型分析文本的说服力特征
用法:
    python data/preprocess.py --input raw_data/ECTF/train.csv --output pcot_persu_data/deepseek-v3.2_ECTF_train.csv --model deepseek-v3.2
"""

import os
import sys
import argparse
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# 加载环境变量
load_dotenv()

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", None)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY_")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", None)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", None)


def get_client(model: str):
    """根据模型名称获取API客户端"""
    if model in ["deepseek-reasoner", "deepseek-chat", "deepseek-v3.2"]:
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
    elif "gpt" in model:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
    elif "claude" in model:
        return OpenAI(
            api_key=CLAUDE_API_KEY,
            base_url=CLAUDE_BASE_URL
        )
    elif "gemini" in model:
        return OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


def call_model(client, model: str, text: str) -> str:
    """调用大模型获取说服力分析结果"""
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_single_row(args):
    """处理单行数据"""
    index, text, model = args
    client = get_client(model)
    result = call_model(client, model, text)
    return index, result


def preprocess_dataset(input_file: str, output_file: str, model: str = "deepseek-chat", max_workers: int = 10):
    """
    预处理数据集

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        model: 使用的模型名称
        max_workers: 并发线程数
    """
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")

    # 确保有content列
    if 'content' not in df.columns:
        possible_cols = ['text', 'sentence', 'statement', 'news']
        for col in possible_cols:
            if col in df.columns:
                df['content'] = df[col]
                break
        else:
            raise ValueError(f"Cannot find content column in {input_file}")

    # 添加新列
    df['system_prompt'] = SYSTEM_PROMPT
    df['user_prompt'] = USER_PROMPT_TEMPLATE.format(text="[text]")
    df['generated_pred'] = None

    # 并发处理
    tasks = [(idx, row['content'], model) for idx, row in df.iterrows()]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_row, task): task[0] for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                idx, result = future.result()
                if result:
                    results[idx] = result
            except Exception as e:
                print(f"Error processing row: {e}")

    # 更新DataFrame
    for idx, result in results.items():
        df.at[idx, 'generated_pred'] = result
        df.at[idx, 'user_prompt'] = USER_PROMPT_TEMPLATE.format(text=df.at[idx, 'content'])

    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理数据集，使用大模型分析说服力特征")
    parser.add_argument("--input", type=str, required=True, help="输入CSV文件")
    parser.add_argument("--output", type=str, required=True, help="输出CSV文件")
    parser.add_argument("--model", type=str, default="deepseek-v3.2", help="使用的模型")
    parser.add_argument("--max_workers", type=int, default=10, help="并发线程数")
    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.model, args.max_workers)