# 数据工程模块

这是 **CSInterviewAgent_Qwen3** 项目的数据工程部分。该模块负责处理原始面试题数据，并生成用于模型微调（SFT）、知识图谱（KG）构建以及 RAG 向量检索的高质量数据资产。

主要功能包括：
* **数据清洗**：从原始 Markdown 面试题中提取结构化 Q&A 对。
* **SFT 数据生成**：基于 LLM 生成“高手 vs 菜鸟”双剧本对话数据。
* **知识图谱构建**：提取计算机领域的技术实体与三元组关系。
* **向量库构建**：基于 ChromaDB 构建本地 RAG 检索库。

## 1. 环境设置

### a. 创建虚拟环境

```bash
conda create -n data_eng python=3.10
conda activate data_eng
```

### b. 安装依赖

```bash
pip install -r requirements.txt
# 验证安装 (确保已安装 OpenAI SDK 和 LangChain 相关组件)
python -c "import openai; import langchain; print('Environment setup successful.')"
```

**注意**：本项目依赖 SiliconFlow (硅基流动) 的 API 服务。请在运行脚本前，确保 generate_data.py, generate_kg.py 和 rag_demo.py 文件中的 API_KEY 已替换为您自己的密钥。

## 2. 数据处理流程
### 步骤 A：原始数据清洗
原始数据来源于0voice开源数据https://github.com/0voice/interview_internal_reference

将原始 Markdown 面试题文件放置在 `0voice/ALL` 目录（或脚本中配置的路径），运行清洗脚本：

```bash
python preprocess.py
```

* **输入**：原始 Markdown 文件。
* **输出**：`cleaned_data/cleaned_data.json` (结构化的 Q&A 列表)。
* **说明**：脚本使用正则+状态机算法提取题目与答案，去除无关字符。

### 步骤 B：双剧本微调数据生成 (SFT)
调用Qwen3-14B模型，将单轮QA扩充为“高手版本”和“菜鸟版本”的多轮评估对话。

```bash
python generate_data.py
```

* **输入**：`cleaned_data/cleaned_data.json`
* **输出**：`finetune_data/interview_finetune_data.json`
* **说明**：此过程调用 LLM API，耗时较长（约 8 小时），生成的 JSON 包含 Expert (高手) 和 Struggle (菜鸟) 两种剧本。

### 步骤 C：知识图谱提取 (KG)
从文本中提取技术实体及其三元组关系 (Head, Relation, Tail)。

```bash
python generate_kg.py
```

* **输入**：`cleaned_data/cleaned_data.json`
* **输出**：`kg_data/knowledge_graph.json`
* **说明**：包含黑名单过滤机制与自环检测，以确保图谱质量。

### 步骤 D：RAG 向量库构建与测试
使用 Pro/BAAI/bge-m3 模型生成向量索引，并存入 ChromaDB。

```bash
python rag_demo.py
```

* **输入**：`cleaned_data/cleaned_data.json`
* **输出**：`my_vector_db/` 目录 (包含 .bin 和 .sqlite3 文件)。
* **功能**：脚本运行后会构建数据库（如不存在），并执行一次简单的关键词检索测试以验证环境。

## 3. 实验配置与参数

本模块主要依赖云端 API，核心配置位于各 Python 脚本头部。以下是关键参数说明：

### a. 双剧本生成配置 (`generate_data.py`)
* **Model**: `Qwen/Qwen3-14B`
* **Temperature**: `0.7` (保证对话多样性)
* **Prompt策略**: 双层提示词 (System: 格式约束 / User: 剧本设定)
* **预计消耗**: 约 4000K Tokens / 8 小时

### b. 知识图谱提取配置 (`generate_kg.py`)
* **Model**: `Qwen/Qwen3-14B`
* **Temperature**: `0.1` (保证提取严谨性)
* **过滤机制**: 包含黑名单过滤与自环检测
* **预计消耗**: 包含在上述 4000K Tokens 中 / 2 小时

### c. 向量检索配置 (`rag_demo.py`)
* **Embedding Model**: `Pro/BAAI/bge-m3`
* **Vector DB**: ChromaDB (Local Mode)
* **预计消耗**: 约 600K Tokens / 20 分钟

## 4. 文件结构

```
data_engineering/
├── cleaned_data/
│   └── cleaned_data.json            # 清洗后的结构化问答数据
├── finetune_data/
│   └── interview_finetune_data.json # 用于微调的双剧本SFT数据
├── kg_data/
│   └── knowledge_graph.json         # 提取的三元组知识图谱
├── my_vector_db/                    # RAG向量数据库目录
│   ├── e5dd9e.../                   # (UUID目录) HNSW 索引二进制文件
│   │   ├── data_level0.bin          # 底层向量数据
│   │   ├── header.bin               # 索引头配置
│   │   ├── link_lists.bin           # 图结构邻接表
│   │   └── ...                    
│   └── chroma.sqlite3               # 向量库元数据 (SQLite)
├── generate_data.py                 # [脚本] 执行双剧本微调数据生成
├── generate_kg.py                   # [脚本] 执行知识图谱三元组提取
├── preprocess.py                    # [脚本] 执行原始Markdown数据清洗
├── rag_demo.py                      # [脚本] RAG 向量库构建与检索演示
└── README.md                        # 本说明文档
```


