import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback # 【修改1】引入 TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab
import matplotlib.pyplot as plt

# ================== 1. 核心配置区域 ==================

# 【A】路径配置
ROOT_DIR = "/opt/data/private/qwen3_train"
DATA_PATH = os.path.join(ROOT_DIR, "interview_data.json")
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model_cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
SWANLAB_DIR = os.path.join(ROOT_DIR, "swanlog")

# 确保目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SWANLAB_DIR, exist_ok=True)

# 【B】模型与Prompt配置
MODEL_ID = "Qwen/Qwen3-8B"
MAX_LENGTH = 4096

# 定义全局统一的 System Prompt
SYSTEM_PROMPT = "你是一位专业的计算机专业面试官，风格严谨，喜欢追问底层原理。请根据候选人的回答进行追问或点评。面试中对话不超过10轮，完成面试时面试官主动结束并给出打分和点评。"

# 初始化 SwanLab
os.environ["SWANLAB_PROJECT"] = "qwen3-8b-interview-sft"
swanlab.init(project=os.environ["SWANLAB_PROJECT"], mode="local", logdir=SWANLAB_DIR)

swanlab.config.update({
    "model": MODEL_ID,
    "data_path": DATA_PATH,
    "data_max_length": MAX_LENGTH,
    "method": "LoRA",
    "task": "Interviewer Simulation",
    "system_prompt": SYSTEM_PROMPT
})


# ================== 2. 数据处理函数 ==================
def process_func(example):
    input_ids = []
    labels = []

    conversation = example["conversations"]

    if conversation[0]["from"] != "system":
        system_head = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
        system_content = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)
        system_tail = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

        input_ids += system_head + system_content + system_tail
        labels += [-100] * (len(system_head) + len(system_content) + len(system_tail))

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}

    for message in conversation:
        role = message["from"]
        content = message["value"]
        qwen_role = role_map.get(role, "user")

        head_text = f"<|im_start|>{qwen_role}\n"
        head_ids = tokenizer.encode(head_text, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        tail_text = "<|im_end|>\n"
        tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)

        current_ids = head_ids + content_ids + tail_ids
        input_ids.extend(current_ids)

        if role == "gpt":
            current_labels = [-100] * len(head_ids) + content_ids + tail_ids
        else:
            current_labels = [-100] * len(current_ids)

        labels.extend(current_labels)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ================== 3. 模型下载与加载 ==================
print(f"正在下载/加载模型: {MODEL_ID}...")
print(f"缓存目录: {MODEL_CACHE_DIR}")

try:
    model_dir = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR, revision="master")
except Exception as e:
    print(f"下载报错: {e}")
    raise e

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

try:
    if model.generation_config:
        setattr(model.generation_config, "enable_thinking", True)
except Exception:
    pass

model.enable_input_require_grads()

# ================== 4. LoRA 配置 ==================
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# ================== 5. 数据集加载 ==================
print(f"正在读取数据: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"错误：找不到文件 {DATA_PATH}，请确认文件名正确！")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

full_ds = Dataset.from_list(data_list)
split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]

train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# ================== 【修改2】定义回调函数 ==================
class EvalAtStep10Callback(TrainerCallback):
    """
    自定义回调：仅在第 10 步时强制触发一次评估
    """
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 10:
            control.should_evaluate = True

# ================== 6. 训练参数 ==================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    num_train_epochs=2,  # 【修改3】改为 2 轮
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-interview",
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[EvalAtStep10Callback()] # 【修改4】注册回调
)

print("🚀 开始训练面试官模型...")
trainer.train()

# ================== 【Loss 绘图部分 (无需修改，会自动读取)】 ==================
print("📊 正在生成 Loss 曲线图...")

# 提取日志历史
log_history = trainer.state.log_history

# 分离训练 loss 和验证 loss
train_steps = []
train_loss = []
eval_steps = []
eval_loss = []

for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
    if "eval_loss" in log and "step" in log:
        eval_steps.append(log["step"])
        eval_loss.append(log["eval_loss"])

# 绘图
plt.figure(figsize=(10, 6))

if train_steps:
    plt.plot(train_steps, train_loss, label="Training Loss", alpha=0.7, color="blue")

if eval_steps:
    plt.plot(eval_steps, eval_loss, label="Evaluation Loss", marker='o', color="red", linestyle="--")

plt.xlabel("Global Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss Curve")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_plot_path)
print(f"✅ Loss 曲线已保存至: {loss_plot_path}")
plt.close()

# ================== 7. 训练后测试 ==================
print("=== 开始模拟面试测试 ===")
test_samples = eval_ds.select(range(min(2, len(eval_ds))))
test_text_list = []

for sample in test_samples:
    conversations = sample['conversations']
    input_messages = []
    ground_truth = ""
    last_human_idx = -1

    for i, msg in enumerate(conversations):
        if msg['from'] == 'human':
            last_human_idx = i
        elif msg['from'] == 'gpt' and i > last_human_idx:
            ground_truth = msg['value']

    input_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    for i in range(last_human_idx + 1):
        msg = conversations[i]
        role = "user" if msg['from'] == "human" else "assistant"
        input_messages.append({"role": role, "content": msg['value']})

    response = predict(input_messages, model, tokenizer)
    last_user_input = input_messages[-1]['content']

    log_text = f"""
    【Context System】: {input_messages[0]['content']}
    【Candidate Answer】: {last_user_input}
    【Real Interviewer】: {ground_truth}
    【AI Interviewer】: {response}
    """

    test_text_list.append(swanlab.Text(log_text))
    print(log_text)
    print("-" * 50)

swanlab.log({"Interview_Prediction": test_text_list})
swanlab.finish()