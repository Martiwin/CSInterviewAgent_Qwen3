import os
import json
import time
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings 

# ================= 配置区域 =================
API_KEY = "sk-cskqfbenfwkrqghbqjiljkqxlnuzfzbqtdtpwezowbkgpcmi" 
BASE_URL = "https://api.siliconflow.cn/v1"
INPUT_JSON = "cleaned_data/cleaned_data.json"
DB_PATH = "my_vector_db_v2"
# ===========================================


def get_embedding_model():
    return OpenAIEmbeddings(
        model="Pro/BAAI/bge-m3", 
        openai_api_key=API_KEY, 
        openai_api_base=BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=32 
    )


def build_vector_db():
    print("正在加载清洗后的 JSON 数据...")
    
    if not os.path.exists(INPUT_JSON):
        print(f"找不到 {INPUT_JSON}，请先运行数据清洗脚本。")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        raw_qa_list = json.load(f)

    print(f"加载了 {len(raw_qa_list)} 个知识点。正在转换为向量文档...")

    documents = []
    for item in raw_qa_list:
        # 组合问题和答案，作为检索内容
        text_content = f"面试题：{item['topic']}\n标准答案：{item['content']}"
        
        doc = Document(
            page_content=text_content,
            metadata={"source": item['origin_file'], "topic": item['topic']}
        )
        documents.append(doc)

    print(f"正在调用云端 API (BAAI/bge-m3) 生成向量...")
    embedding_model = get_embedding_model()
    
    try:
        # 批量写入 Chroma
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embedding_model, 
            persist_directory=DB_PATH
        )
        print("向量库构建完成！")
    except Exception as e:
        print(f"构建失败: {e}")


def query_rag(query_text):
    print(f"\n正在检索：{query_text}")
    embedding_model = get_embedding_model()
    
    if not os.path.exists(DB_PATH):
        print("向量库不存在，请先运行 build_vector_db()")
        return []

    try:
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        results = vectorstore.similarity_search(query_text, k=3)
    except Exception as e:
        print(f"检索失败: {e}")
        return []
    
    print("-" * 30)
    if results:
        for i, res in enumerate(results):
            source = res.metadata.get('source', '未知')
            content_preview = res.page_content[:150].replace('\n', ' ')
            print(f"[参考资料 {i+1}] (来源: {source}):\n{content_preview}...\n")
    else:
        print("未找到相关资料")
    return results


if __name__ == "__main__":
    # build_vector_db()  # 第一次运行需要建库

    query_rag("引用")
