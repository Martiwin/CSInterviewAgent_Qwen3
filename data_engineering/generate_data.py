import os
import json
import time
import re
from urllib import response
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-cskqfbenfwkrqghbqjiljkqxlnuzfzbqtdtpwezowbkgpcmi"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-14B"
INPUT_JSON = "cleaned_data/cleaned_data.json"
# 输出文件
OUTPUT_FILE = "finetune_data/interview_finetune_data_v3.json"

# 限制处理数量
MAX_ITEMS_TO_PROCESS = 9999
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_cleaned_data(file_path):
    """直接读取清洗好的 JSON 文件"""
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        print("请先运行 export_clean_data.py 生成清洗后的数据！")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"成功加载 {len(data)} 条清洗后的数据。")
    return data


def generate_dialogue_dual_version(qa_data):
    results = []
    print(f"开始调用 {MODEL_NAME} 生成带【评分功能】的对话...")
    
    # 截取前 N 条进行处理
    target_data = qa_data[:MAX_ITEMS_TO_PROCESS]
    
    for item in tqdm(target_data):
        topic = item['topic']
        content = item['content']
        
        # 防止 Token 溢出
        if len(content) > 1500:
            content = content[:1500] + "..."

        # 定义两个场景
        scenarios = [
            {
                "type": "expert",
                "score_range": "85-95分",
                "verdict": "通过",
                "desc": "【剧本A：高手过招】\n候选人回答准确。面试官进行深度追问后表示满意。\n最后面试官给出高分，并称赞其底层原理扎实。"
            },
            {
                "type": "struggle",
                "score_range": "50-65分",
                "verdict": "不通过或待定",
                "desc": "【剧本B：基础薄弱】\n候选人回答吞吞吐吐或有错误。面试官尝试引导但效果一般。\n最后面试官给出低分，并委婉指出其基础概念需要加强。"
            }
        ]

        for scenario in scenarios:
            prompt = f"""
            你是一个构建微调数据的专家。请将面试题改编成一段【完整的、包含评分环节的】多轮面试对话。
            
            【面试素材】
            题目：{topic}
            参考答案：{content}
            
            【剧本要求】：{scenario['desc']}

            【对话结构流程】（请严格遵守）：
            1. **开场 (Round 1)**: 
               - Human: "面试官你好，我准备好了。"
               - AI: "你好，请简单做一个自我介绍，包括你的目标岗位。"
            
            2. **自我介绍 & 提问 (Round 2)**:
               - Human: (基于题目内容生成简短自我介绍) "我是xx，应聘xx岗位..."
               - AI: "好的。那我们直接开始。请问..." (抛出题目)
            
            3. **追问环节 (Round 3-5)**: 
               - 包含 2-5 轮来回。AI 针对 User 的回答进行追问（Socratic Method）。
            
            4. **结束 & 评分 (Final Round)**:
               - AI 必须主动结束面试。
               - **关键要求**：AI 的最后一句回复必须包含【面试总结】。
               - 总结格式要求：
                 "好的，今天的面试就到这里。
                 【面试评分】：{scenario['score_range']}（请生成一个具体数字）
                 【面试评价】：(一句话点评亮点或不足)
                 【最终结果】：{scenario['verdict']}"

            【输出格式】：
            直接输出 JSON 对象：
            {{
                "conversations": [
                    {{"from": "human", "value": "..."}},
                    {{"from": "gpt", "value": "..."}}
                ]
            }}
            """

            # === 增加重试机制，解决 JSON 解析报错 ===
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "你是一个严谨的数据生成助手，只输出JSON。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=2500,
                        response_format={"type": "json_object"}
                    )
                    
                    raw_text = response.choices[0].message.content.strip()
                    
                    # === 增加正则提取（双重保险）===
                    # 有时候模型会在 JSON 前后加废话，用正则只提取 { ... }
                    json_match = re.search(r'\{[\s\S]*\}', raw_text)
                    if json_match:
                        clean_json_str = json_match.group()
                    else:
                        clean_json_str = raw_text

                    data = json.loads(clean_json_str)
                    
                    if "conversations" in data:
                        data["id"] = f"identity_{int(time.time()*10000)}_{scenario['type']}"
                        results.append(data)
                        break # 成功了就跳出重试循环
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"生成失败 ({scenario['type']}) - {e}")
                    else:
                        time.sleep(0.5)
                    continue

    return results

if __name__ == "__main__":
    # 读取 JSON
    raw_qa = load_cleaned_data(INPUT_JSON)
    
    if raw_qa:
        # 生成微调数据
        final_data = generate_dialogue_dual_version(raw_qa)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        # 保存结果
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        print(f"全部生成完毕！共 {len(final_data)} 条高质量对话。")
        print(f"文件已保存至: {OUTPUT_FILE}")
