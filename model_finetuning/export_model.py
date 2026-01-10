import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope import snapshot_download
import os
import shutil

# ================= 配置区域 =================
ROOT_DIR = "/opt/data/private/qwen3_train"
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model_cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MERGED_DIR = os.path.join(ROOT_DIR, "qwen3_interview_merged") # 合并后存放的临时目录
MODEL_ID = "Qwen/Qwen3-8B"
# ===========================================

def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])

# 1. 获取路径
print("🔍 正在定位路径...")
base_model_path = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR, revision="master")
lora_path = get_latest_checkpoint(OUTPUT_DIR)
print(f"✅ 基础模型: {base_model_path}")
print(f"✅ LoRA权重: {lora_path}")

# 2. 加载并合并
print("⏳ 正在加载并合并模型 (这可能需要几分钟)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, # 转换 GGUF 建议先转为 fp16
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload() # 核心：融合权重

# 3. 保存
print(f"💾 正在保存合并后的模型到: {MERGED_DIR}")
if os.path.exists(MERGED_DIR):
    shutil.rmtree(MERGED_DIR) # 清理旧文件
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("🎉 合并完成！准备进行格式转换。")