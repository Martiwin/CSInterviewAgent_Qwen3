import os
import re
import json
from tqdm import tqdm

# ================= 配置区域 =================
SOURCE_FOLDER = "./0voice/ALL"          
JSON_OUTPUT = "cleaned_data.json"       
MAX_FILES = 9999                       
# ===========================================


def clean_text(text):
    """
    清洗文本：
    1. 去除 HTML 标签 (如 <br/>, <p>)
    2. 替换转义符
    3. 去除多余的空行
    """
    # 把 <br>, <br/>, <br /> 替换成换行
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # 去除其他 HTML 标签 (比如 <span>)
    text = re.sub(r'<[^>]+>', '', text)
    # 把连续的 \n 替换成两个 \n (保留段落感但去除过多的空行)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除首尾空白
    return text.strip()


def extract_qa_from_md(folder_path, max_files=5, max_qa_per_file=100):
    qa_list = []
    
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    except FileNotFoundError:
        print(f"错误！找不到文件夹 {folder_path}")
        return []

    print(f"扫描到 {len(files)} 个文件，开始清洗...")
    target_files = files[:max_files]
    
    for filename in tqdm(target_files, desc="处理进度"):
        file_path = os.path.join(folder_path, filename)
        
        current_topic = ""
        current_content = []
        is_in_code_block = False 
        
        # 标记：当前题目是否是由 # 标题触发的
        # 如果是 # 触发的，就不允许数字列表打断它
        topic_started_by_hash = False 
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line in lines:
                stripped = line.strip()
                
                # 代码块保护
                if stripped.startswith("```"):
                    is_in_code_block = not is_in_code_block
                    current_content.append(line)
                    continue
                if is_in_code_block:
                    current_content.append(line)
                    continue

                # ================= 状态特征识别 =================
                is_hash_header = stripped.startswith('#') 
                
                # 黑名单
                ignore_keywords = ["答案", "参考", "解析", "出题人", "专家", "来源", "链接"]
                is_ignore_header = any(k in stripped for k in ignore_keywords)
                
                # 数字列表 (1. xxx)
                is_numbered_list = re.match(r'^[\d]+\s*[\.、]\s+.*', stripped)
                
                # ================= 核心判定逻辑 =================
                
                is_new_topic = False
                
                # 遇到 # 标题，且不是干扰词 -> 绝对的新题
                if is_hash_header and not is_ignore_header:
                    is_new_topic = True
                    topic_started_by_hash = True
                
                # 遇到数字列表，且当前题目不是由 # 开头的 -> 认为是新题
                # 如果当前题已经是 # 开头的，那这个数字列表大概率是答案的一部分，忽略
                elif is_numbered_list and not topic_started_by_hash and len(stripped) < 50:
                    is_new_topic = True
                    topic_started_by_hash = False  # 数字开头的题，随时可能被下一个数字打断
                
                # ================= 分支处理 =================
                if is_new_topic:
                    # 保存上一题
                    if current_topic and current_content:
                        clean_topic = re.sub(r'^[#\d\.\s]+', '', current_topic)
                        clean_topic = clean_topic.replace("问题：", "").replace("题目：", "").replace("**", "").strip()
                        
                        # 拼接内容并清洗 HTML
                        raw_body = "".join(current_content)
                        clean_body = clean_text(raw_body)
                        
                        if len(clean_body) > 2 and len(clean_topic) > 2:
                            current_file_count = len([x for x in qa_list if x['origin_file'] == filename])
                            if current_file_count < max_qa_per_file:
                                qa_list.append({
                                    "origin_file": filename,
                                    "topic": clean_topic,
                                    "content": clean_body
                                })
                    
                    # 开启新题
                    current_topic = stripped
                    current_content = [] 
                
                else:
                    # 处理内容部分
                    if is_hash_header and any(k in stripped for k in ["出题人", "专家", "来源", "链接"]):
                        continue
                        
                    if is_hash_header and any(k in stripped for k in ["答案", "参考", "解析"]):
                        # 提取同一行的答案
                        cleaned_line = re.sub(r'^[#\s]+', '', stripped)
                        cleaned_line = cleaned_line.replace("**", "").replace("参考答案", "").replace("答案", "").replace("参考", "").replace("：", "").replace(":", "").strip()
                        if cleaned_line:
                            current_content.append(cleaned_line + "\n")
                        continue

                    current_content.append(line)

            # 保存最后一题
            if current_topic and current_content:
                clean_topic = re.sub(r'^[#\d\.\s]+', '', current_topic)
                clean_topic = clean_topic.replace("问题：", "").replace("题目：", "").replace("**", "").strip()
                
                raw_body = "".join(current_content)
                clean_body = clean_text(raw_body)
                
                current_file_count = len([x for x in qa_list if x['origin_file'] == filename])
                if len(clean_body) > 2 and current_file_count < max_qa_per_file:
                    qa_list.append({
                        "origin_file": filename,
                        "topic": clean_topic,
                        "content": clean_body
                    })

        except Exception as e:
            print(f"解析 {filename} 时出错: {e}")

    return qa_list


if __name__ == "__main__":
    data = extract_qa_from_md(SOURCE_FOLDER, max_files=MAX_FILES, max_qa_per_file=9999)
    
    if not data:
        print("没有提取到数据")
    else:
        with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"全部完成！<br> 等 HTML 标签已自动清洗。")
