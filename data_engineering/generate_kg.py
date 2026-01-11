import os
import json
import re
import time
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-cskqfbenfwkrqghbqjiljkqxlnuzfzbqtdtpwezowbkgpcmi"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-14B" 

INPUT_JSON = "cleaned_data/cleaned_data.json"
OUTPUT_FILE = "kg_data/knowledge_graph.json"

# 限制处理数量
MAX_ITEMS_TO_PROCESS = 9999

# 黑名单
BLACKLIST_ENTITIES = {
    # --- 通用泛指词 ---
    "程序", "系统", "软件", "应用", "用户", "客户", "功能", "方法", 
    "数据", "信息", "内容", "问题", "答案", "时间", "机器", "设备", 
    "它", "他们", "我们", "之一", "一方面", "例子", "优点", "缺点",
    "技术", "环境", "场景", "情况", "部分", "整体", "特点", "原理",
    
    # --- 动作/过程描述 (非实体) ---
    "操作", "通知", "变化", "状态", "结果", "过程", "方式", "步骤",
    "发送", "接收", "读写", "访问", "修改", "删除", "创建", "处理",
    
    # --- 编程语言通用词 (Java/C++) ---
    "对象", "类", "接口", "变量", "参数", "代码", "函数", "属性", "字段",
    "实现", "继承", "多态", "逻辑", "空", "null", "true", "false",
    
    # --- 操作系统/网络通用词 ---
    "空间", "模式", "内核", "文件", "目录", "硬盘", "内存", "网络",
    "连接", "请求", "响应", "协议", "端口", "地址", "消息", "包",
    "性能", "效率", "速度", "开销", "资源", "瓶颈"
}
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_cleaned_data(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功加载 {len(data)} 条数据。")
    return data


def is_valid_triple(triple):
    """
    严格校验三元组质量
    """
    h = triple.get('head', '').strip()
    t = triple.get('tail', '').strip()
    r = triple.get('relation', '').strip()
    
    if not h or not t or not r: return False
    
    # 长度过滤
    if len(h) < 2 or len(t) < 2: return False
    if len(h) > 15 or len(t) > 15: return False # 太长通常是句子
    
    # 黑名单过滤
    if h in BLACKLIST_ENTITIES or t in BLACKLIST_ENTITIES: return False
    
    # 循环过滤
    if h == t: return False
    
    # 包含过滤 (防止 "Redis" -> 包含 -> "Redis持久化")
    if h in t or t in h:
         # 允许部分包含，但不能太像。这里简单处理，若高度重合则丢弃
         if len(set(h) & set(t)) / len(set(h) | set(t)) > 0.8:
             return False

    return True


def extract_knowledge_triples(qa_data):
    kg_data = []
    print(f"开始构建知识图谱 (前 {MAX_ITEMS_TO_PROCESS} 条)...")
    
    target_data = qa_data[:MAX_ITEMS_TO_PROCESS]
    
    for item in tqdm(target_data):
        full_text = f"问题：{item['topic']}\n答案：{item['content']}"
        if len(full_text) > 2000: full_text = full_text[:2000]

        prompt = f"""
        任务：你是一位精通操作系统、数据库、Java、C++、分布式架构等计算机各个方面的【全栈技术专家】。请从文本中构建高精度的【技术概念知识图谱】。
        
        【待分析文本】：
        {full_text}
        
        【提取法则】（违者必究）：
        1. **实体必须是具体的计算机方面的专有名词**：
           - 优选：Epoll, 红黑树, 零拷贝, MVCC, HashMap, CAS, 用户态, 页缓存等计算机方面的技术概念名词
           - 剔除：数据, 效率, 方式, 步骤, 事情, 地方, 东西等等
           
        2. **关系必须表达具体的【技术原理】或【架构归属】**：
           - 优选：
             - 属于/分类 (e.g. ArrayList --属于--> List集合)
             - 包含/组成 (e.g. JVM内存 --包含--> 堆区)
             - 底层结构 (e.g. Redis Zset --底层结构--> 跳表)
             - 核心特性 (e.g. TCP --保证--> 可靠性)
             - 导致/解决 (e.g. 死锁 --导致--> 系统卡死)
           - 拒绝【动作描述】：
             - 发送, 接收, 看见, 告诉, 变成, 使得

        3. **处理"如何/怎么"类问题**：
           - 遇到 "select如何实现"，请提取 "Select机制" --基于--> "轮询" 或 "Select" --限制--> "1024连接"。
        
        【输出格式】：
        {{
            "triples": [
                {{"head": "实体1", "relation": "关系", "tail": "实体2"}}
            ]
        }}
        """

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个严谨的知识提取助手，只输出JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1, 
                    response_format={"type": "json_object"} 
                )
                
                raw_text = response.choices[0].message.content.strip()
                json_match = re.search(r'\{[\s\S]*\}', raw_text)
                clean_json_str = json_match.group() if json_match else raw_text

                data = json.loads(clean_json_str)
                
                if "triples" in data and isinstance(data["triples"], list):
                    for triple in data["triples"]:
                        if is_valid_triple(triple):
                            triple["source_topic"] = item["topic"]
                            kg_data.append(triple)
                    break 
                
            except Exception:
                time.sleep(0.5)
                continue
    return kg_data


if __name__ == "__main__":
    raw_data = load_cleaned_data(INPUT_JSON)
    if raw_data:
        triples = extract_knowledge_triples(raw_data)
        if triples:
            os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(triples, f, ensure_ascii=False, indent=2)
            
            print(f"知识图谱构建完成！共提取 {len(triples)} 个三元组。")
            
            print("\n=== 效果抽查 ===")
            for t in triples[:10]:
                print(f"{t['head']} --[{t['relation']}]--> {t['tail']}")
