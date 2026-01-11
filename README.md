# CSInterviewAgent_Qwen3
当前通用大模型在模拟计算机行业面试时存在明显不足：缺乏深度追问的逻辑性、底层代码回答容易出现“幻觉”、语气过于客气且无法营造真实压力面试环境。为应对这些问题，我们基于覆盖计算机各领域的面试数据集，分别对Qwen3进行了SFT微调、在未微调的Qwen3上使用RAG检索增强和知识图谱等技术，打造了两版风格严厉、回答规范且具备关联追问能力的面试官。

## 项目文件结构

```
CSInterviewAgent_Qwen3/
├── agents/                            # 智能体运行子系统
│   ├── interviewer-controller/
│   │   ├── src/main/java/com/example/interviewer_controller/
│   │   │   ├── config/                # Spring Bean 配置 (Chroma注入)
│   │   │   ├── controller/            # REST API 接口 (语音/文字)
│   │   │   ├── service/               # 核心业务逻辑
│   │   │   │   ├── InterviewService.java      # 状态机与 RAG 调度
│   │   │   │   ├── GraphKnowledgeService.java # 知识图谱导航
│   │   │   │   ├── SpeechService.java         # TTS/STT 容器通信
│   │   │   │   └── InterviewSession.java      # 会话状态管理
│   │   │   └── model/                 # 知识图谱数据实体
│   │   └── src/main/resources/
│   │       ├── static/                # 前端 Web 交互界面
│   │       └── application.properties # 系统全局配置
│   ├── Qwen3_model/
│   │   ├── Modelfile                  # 微调模型定义文件
│   │   └── qwen3_interview_q4_k_m.gguf # 微调后的 GGUF 模型权重
│   ├── tts_module/
│   │   ├── Dockerfile                 # Edge-TTS 容器构建文件
│   │   ├── requirements.txt           # Python 依赖声明
│   │   └── tts_server.py              # TTS 服务实现脚本
│   └── README.md                      # 环境配置与运行说明
│
├── data_engineering/                  # 数据工程模块
│   ├── cleaned_data/
│   │   └── cleaned_data.json          # 清洗后的结构化问答数据
│   ├── finetune_data/
│   │   └── interview_finetune_data.json # 生成的双剧本SFT数据
│   ├── kg_data/
│   │   └── knowledge_graph.json       # 提取的三元组知识图谱
│   ├── my_vector_db/                  # RAG向量数据库目录
│   │   ├── e5dd9e.../                 # (UUID目录) HNSW 索引二进制文件
│   │   │   ├── data_level0.bin        # 底层向量数据
│   │   │   ├── header.bin             # 索引头配置
│   │   │   ├── link_lists.bin         # 图结构邻接表
│   │   │   └── ...                    
│   │   └── chroma.sqlite3             # 向量库元数据 (SQLite)
│   ├── generate_data.py               # 执行双剧本微调数据生成
│   ├── generate_kg.py                 # 执行知识图谱三元组提取
│   ├── preprocess.py                  # 执行原始Markdown数据清洗
│   ├── rag_demo.py                    # RAG 向量检索演示脚本
│   ├── requirements.txt               # 依赖项
│   └── README.md                      # 环境配置与运行说明
│
├── model_finetuning/                  # 模型微调代码
│   ├── chat.py
│   ├── compare_models.py
│   ├── export_model.py
│   ├── train.py
│   ├── chat.png
│   ├── comparison_chart.png
│   ├── loss_curve.png
│   └── README.md
│
└── README.md
```
