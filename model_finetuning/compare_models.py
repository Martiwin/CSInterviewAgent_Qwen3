import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download
import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ================= 1. 配置区域 =================
ROOT_DIR = "/opt/data/private/qwen3_train"
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model_cache")
DATA_PATH = os.path.join(ROOT_DIR, "interview_data.json")
MODEL_ID = "Qwen/Qwen3-8B"

# 【重要】测试样本数限制
# 设为 30 大约跑 10分钟；设为 None 则跑完所有数据(约1小时)
TEST_SAMPLE_NUM = None

# System Prompt
SYSTEM_PROMPT = "你是一位专业的计算机专业面试官，风格严谨，喜欢追问底层原理。请根据候选人的回答进行追问或点评。面试中对话不超过10轮，完成面试时面试官主动结束并给出打分和点评。"


# ==============================================

def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])


def load_model_and_tokenizer():
    print("⏳ 正在加载模型 (这只需要加载一次)...")
    try:
        model_dir = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR, revision="master")
    except:
        model_dir = MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 1. 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 2. 挂载 LoRA
    lora_path = get_latest_checkpoint(OUTPUT_DIR)
    if not lora_path:
        raise FileNotFoundError("❌ 未找到 output 下的 checkpoint，请先运行 train.py")

    print(f"✅ 挂载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, history):
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def calculate_metrics(predictions, references):
    rouge = Rouge()
    smooth = SmoothingFunction().method1
    bleu_scores = []
    rouge_l_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred) if pred else [" "]
        ref_tokens = list(ref) if ref else [" "]

        if not pred.strip(): pred = " "

        # BLEU-4
        score = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        bleu_scores.append(score)

        # ROUGE-L
        try:
            scores = rouge.get_scores(" ".join(pred_tokens), " ".join(ref_tokens))
            rouge_l_scores.append(scores[0]['rouge-l']['f'])
        except:
            rouge_l_scores.append(0.0)

    return {
        "BLEU-4": np.mean(bleu_scores),
        "ROUGE-L": np.mean(rouge_l_scores)
    }


def plot_comparison(base_metrics, sft_metrics, save_path):
    labels = ['BLEU-4', 'ROUGE-L']
    base_scores = [base_metrics['BLEU-4'], base_metrics['ROUGE-L']]
    sft_scores = [sft_metrics['BLEU-4'], sft_metrics['ROUGE-L']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    rects1 = plt.bar(x - width / 2, base_scores, width, label='Base Model', color='#d3d3d3')
    rects2 = plt.bar(x + width / 2, sft_scores, width, label='Fine-tuned (Ours)', color='#4e79a7')

    plt.ylabel('Score')
    plt.title('Performance Comparison: Base vs Fine-tuned')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()

    plt.bar_label(rects1, padding=3, fmt='%.2f')
    plt.bar_label(rects2, padding=3, fmt='%.2f')

    plt.savefig(save_path, dpi=300)
    print(f"📊 对比图表已保存: {save_path}")


def main():
    # 1. 准备数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_data = data[int(len(data) * 0.9):]  # 取后10%

    if TEST_SAMPLE_NUM and len(test_data) > TEST_SAMPLE_NUM:
        print(f"⚠️ 仅截取前 {TEST_SAMPLE_NUM} 条数据进行快速对比...")
        test_data = test_data[:TEST_SAMPLE_NUM]

    model, tokenizer = load_model_and_tokenizer()

    results = []
    preds_base = []
    preds_sft = []
    ground_truths = []

    print("🚀 开始双模型推理对比...")
    for item in tqdm(test_data):
        convs = item['conversations']

        last_human_idx = -1
        for i, msg in enumerate(convs):
            if msg['from'] == 'human':
                last_human_idx = i

        if last_human_idx == -1: continue

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i in range(last_human_idx + 1):
            role = "user" if convs[i]['from'] == "human" else "assistant"
            history.append({"role": role, "content": convs[i]['value']})

        ground_truth = convs[last_human_idx + 1]['value']
        ground_truths.append(ground_truth)

        # --- 核心逻辑：分别推理 ---

        # 1. Base Model (临时禁用 Adapter)
        with model.disable_adapter():
            res_base = predict(model, tokenizer, history)
            preds_base.append(res_base)

        # 2. SFT Model (正常启用 Adapter)
        res_sft = predict(model, tokenizer, history)
        preds_sft.append(res_sft)

        results.append({
            "User Query": history[-1]['content'],
            "Ground Truth": ground_truth,
            "Base Prediction": res_base,
            "SFT Prediction": res_sft
        })

    # 2. 计算指标
    print("📈 正在计算指标...")
    metrics_base = calculate_metrics(preds_base, ground_truths)
    metrics_sft = calculate_metrics(preds_sft, ground_truths)

    print("\n" + "=" * 45)
    print(f"{'Metric':<15} | {'Base Model':<12} | {'SFT Model':<12}")
    print("-" * 45)
    print(f"{'BLEU-4':<15} | {metrics_base['BLEU-4']:.4f}       | {metrics_sft['BLEU-4']:.4f}")
    print(f"{'ROUGE-L':<15} | {metrics_base['ROUGE-L']:.4f}       | {metrics_sft['ROUGE-L']:.4f}")
    print("=" * 45 + "\n")

    # 3. 保存结果
    plot_comparison(metrics_base, metrics_sft, os.path.join(ROOT_DIR, "comparison_chart.png"))

    # 保存 Excel (需要 openpyxl)
    df = pd.DataFrame(results)
    excel_path = os.path.join(ROOT_DIR, "comparison_results.xlsx")
    try:
        df.to_excel(excel_path, index=False)
        print(f"💾 详细对比数据已保存至: {excel_path}")
    except ModuleNotFoundError:
        print("❌ 错误: 未安装 openpyxl，无法保存为 Excel。")
        print("💡 请运行: pip install openpyxl")


if __name__ == "__main__":
    main()