# CSInterviewAgent - 数据工程模块

这是 **CSInterviewAgent_Qwen3** 项目的数据工程部分。该模块负责处理原始面试题数据，并生成用于模型微调（SFT）、知识图谱（KG）构建以及 RAG 向量检索的高质量数据资产。

主要功能包括：
* **数据清洗**：从原始 Markdown 面试题中提取结构化 Q&A 对。
* **SFT 数据生成**：基于 LLM 生成“高手 vs 菜鸟”双剧本对话数据。
* **知识图谱构建**：提取计算机领域的技术实体与三元组关系。
* **向量库构建**：基于 ChromaDB 构建本地 RAG 检索库。

---

## 1. 环境设置

### a. 创建虚拟环境

```bash
conda create -n data_eng python=3.10
conda activate data_eng

### b. 安装依赖

```bash
pip install -r requirements.txt
# 验证安装 (确保已安装 OpenAI SDK 和 LangChain 相关组件)
python -c "import openai; import langchain; print('Environment setup successful.')"




