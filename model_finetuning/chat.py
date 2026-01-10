import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import os
from modelscope import snapshot_download

# ================= 配置区域 (保持与 train.py 一致) =================
ROOT_DIR = "/opt/data/private/qwen3_train"
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model_cache")
MODEL_ID = "Qwen/Qwen3-8B"

# 【关键】必须与 train.py 中的 SYSTEM_PROMPT 完全一致
SYSTEM_PROMPT = "你是一位专业的计算机专业面试官，风格严谨，喜欢追问底层原理。请根据候选人的回答进行追问或点评。面试中对话不超过10轮，完成面试时面试官主动结束并给出打分和点评。"
# ===============================================================

def get_latest_checkpoint(output_dir):
    """自动查找 output 目录下数字最大的 checkpoint 文件夹"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-"):
            try:
                num = int(d.split("-")[-1])
                checkpoints.append((num, os.path.join(output_dir, d)))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]

def load_model():
    print("⏳ 正在寻找最佳微调权重...")
    lora_path = get_latest_checkpoint(OUTPUT_DIR)
    
    if not lora_path:
        print(f"❌ 错误：在 {OUTPUT_DIR} 下没有找到 checkpoint 文件夹！请确认训练是否成功完成。")
        exit()
    print(f"✅ 找到最新权重: {lora_path}")

    print("⏳ 正在定位本地基础模型 (ModelScope)...")
    try:
        # 使用 snapshot_download 获取本地绝对路径，防止重新下载
        model_dir = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR, revision="master")
        print(f"✅ 本地基础模型路径: {model_dir}")
    except Exception as e:
        print(f"❌ 定位基础模型失败: {e}")
        print(f"请检查目录 {MODEL_CACHE_DIR} 是否存在模型文件。")
        exit()

    print("⏳ 正在加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print("⏳ 正在挂载 LoRA 微调权重...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    print("🎉 面试官模型加载完成！")
    return model, tokenizer

def main():
    model, tokenizer = load_model()
    
    # 初始化历史记录
    history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\n" + "="*60)
    print("🤖 AI 面试官已就位。")
    print("💡 指令：输入 'exit' 退出，输入 'clear' 清空记录重新开始。")
    print("="*60 + "\n")

    # 【注意】这里不再抛出预设的开场白，直接进入循环等待用户输入
    
    while True:
        try:
            # 这里的 input 提示符可以简单点，或者留空
            query = input("\nCandidate (你): ")
        except UnicodeDecodeError:
            print("❌ 输入编码错误，请重试")
            continue

        if query.strip() == "":
            continue
        if query.lower() in ["exit", "quit"]:
            print("👋 面试结束。")
            break
        if query.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🔄 面试已重置。")
            continue

        # 加入用户输入
        history.append({"role": "user", "content": query})

        # 构建 Prompt
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7, 
            top_p=0.9,
            repetition_penalty=1.1 
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("Interviewer (AI): ", end="", flush=True)
        response_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response_text += new_text
        print("") 

        history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()