package com.example.interviewer_controller.service;

import com.example.interviewer_controller.model.Triplet;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.InMemoryChatMemory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;
import org.springframework.ai.ollama.api.OllamaOptions;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static org.springframework.ai.chat.client.advisor.AbstractChatMemoryAdvisor.CHAT_MEMORY_CONVERSATION_ID_KEY;
import static org.springframework.ai.chat.client.advisor.AbstractChatMemoryAdvisor.CHAT_MEMORY_RETRIEVE_SIZE_KEY;

@Service
public class InterviewService {

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final GraphKnowledgeService graphService;
    private final Map<String, InterviewSession> sessionStates = new ConcurrentHashMap<>();

    public InterviewService(ChatClient.Builder builder, VectorStore vectorStore, GraphKnowledgeService graphService) {
        this.vectorStore = vectorStore;
        this.graphService = graphService;
        this.chatClient = builder
                .defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory()))
                .build();
    }

//    public String chat(String userText, String sessionId) {
//        System.out.println("\n" + "=".repeat(60));
//        System.out.println("【DEBUG】Session: " + sessionId + " | 用户输入: " + userText);
//
//        InterviewSession session = sessionStates.computeIfAbsent(sessionId, k -> new InterviewSession());
//
//        // --- 阶段 0: 结束判定 ---
//        if (session.isFinished()) {
//            return "本次面试已经结束，谢谢你的参与。请刷新页面重新开始。";
//        }
//
//        // --- 阶段 1: START 握手 ---
//        if ("START".equalsIgnoreCase(userText)) {
//            session.setStep(1);
//            return "你好！我是面试官。请简单介绍下你的技术栈和想面试的岗位。";
//        }
//
//        // --- 阶段 2: 话题锁定与首题抛出 ---
//        if (session.getStep() == 1) {
//            System.out.println("【DEBUG】正在提取话题并生成首题...");
//            // 锁定 3 个话题
//            List<String> allTopics = graphService.getAllTopics();
//            String selectPrompt = String.format("用户介绍：'%s'。从列表%s中选3个最相关的技术话题。只返回名称并用逗号分隔。", userText, allTopics);
//            String rawTopics = chatClient.prompt().user(selectPrompt).call().content();
//            session.setLockedTopics(Arrays.asList(rawTopics.split(",")));
//            session.setCurrentTopicIndex(0);
//            session.setStep(2);
//
//            // 获取首题并加工
//            Document firstDoc = getQuestionFromVectorDB(session.getLockedTopics().get(0));
//            session.setLastExpectedAnswer(firstDoc.getContent());
//
//            String polishedQuestion = polishQuestion(session.getLockedTopics().get(0), firstDoc.getContent());
//
//            System.out.println("【DEBUG】锁定话题: " + session.getLockedTopics());
//            System.out.println("【DEBUG】预存首题答案: " + firstDoc.getContent());
//
//            return polishedQuestion;
//        }
//
//        // --- 阶段 3: 循环评价、动态检索与追问 ---
//        System.out.println("【DEBUG】进入循环评价逻辑...");
//
//        // 1. 评价上轮回答
//        final String lastRefAnswer = session.getLastExpectedAnswer();
//        System.out.println("【DEBUG】参考答案: " + lastRefAnswer);
//
//        // 2. 动态知识图谱检索：根据上轮表现找新知识点
//        // 这里我们把用户回答和上轮答案结合起来搜图谱
//        final String searchKeyword = userText + " " + lastRefAnswer;
//        final String dynamicFacts = graphService.searchFacts(searchKeyword);
//        System.out.println("【DEBUG】动态检索图谱事实: \n" + dynamicFacts);
//
//        // 3. 获取新题目（根据图谱事实或当前话题）
//        Document nextDoc = getQuestionFromVectorDB(session.getLockedTopics().get(session.getCurrentTopicIndex()));
//        session.setLastExpectedAnswer(nextDoc.getContent());
//        System.out.println("【DEBUG】预存本轮答案供下轮用: " + nextDoc.getContent());
//
//        // 4. 判定是否切换话题或结束
//        session.setQuestionCount(session.getQuestionCount() + 1);
//        boolean shouldSwitch = session.getQuestionCount() >= 3; // 每个话题问3道题
//
//        String systemPrompt;
//        if (shouldSwitch && session.getCurrentTopicIndex() >= session.getLockedTopics().size() - 1) {
//            // 面试结束逻辑
//            System.out.println("【DEBUG】判定: 面试结束，准备总结。");
//            session.setFinished(true);
//            systemPrompt = """
//                你是一位面试官。请完成以下任务：
//                1. 评价用户最后一个回答：参考[标准答案]和[图谱事实]。
//                2. 给出一个面试总结：评价用户整体表现（技术广度、深度）。
//                3. 礼貌结束面试。
//
//                [标准答案]: {expectedAnswer}
//                [图谱事实]: {graphFacts}
//                """;
//        } else if (shouldSwitch) {
//            // 切换话题逻辑
//            session.setCurrentTopicIndex(session.getCurrentTopicIndex() + 1);
//            session.setQuestionCount(0);
//            String nextTopicName = session.getLockedTopics().get(session.getCurrentTopicIndex());
//            System.out.println("【DEBUG】判定: 切换话题到 -> " + nextTopicName);
//            systemPrompt = """
//                你是一位面试官。
//                1. 评价用户回答：参考[标准答案]。
//                2. 告诉用户当前话题结束，我们转向下一个话题：""" + nextTopicName + """
//                3. 提出新问题：{nextRawQuestion}（请重新组织语言，问得自然一些）。
//
//                [标准答案]: {expectedAnswer}
//                """;
//        } else {
//            // 继续深挖话题
//            System.out.println("【DEBUG】判定: 继续深挖话题 " + session.getLockedTopics().get(session.getCurrentTopicIndex()));
//            systemPrompt = """
//                你是一位面试官。请组织一段流利的话：
//                1. 评价用户刚才的回答：参考[标准答案]。
//                2. 结合[图谱事实]进行一句话的延伸点评。
//                3. 提出下一个追问：{nextRawQuestion}（请重新组织语言，问得口语化、像真人）。
//
//                [标准答案]: {expectedAnswer}
//                [图谱事实]: {graphFacts}
//                """;
//        }
//
//        final String finalQuestion = extractQuestionOnly(nextDoc.getContent());
//
//        String response = chatClient.prompt()
//                .system(s -> s.text(systemPrompt)
//                        .param("expectedAnswer", lastRefAnswer)
//                        .param("graphFacts", dynamicFacts)
//                        .param("nextRawQuestion", finalQuestion))
//                .user(userText)
//                .advisors(a -> a.param(CHAT_MEMORY_CONVERSATION_ID_KEY, sessionId))
//                .call()
//                .content();
//
//        System.out.println("【DEBUG】面试官回复: " + response);
//        System.out.println("=".repeat(60));
//        return response;
//    }
//
//    // --- 辅助方法 ---
//
//    private String polishQuestion(String topic, String rawContent) {
//        String question = extractQuestionOnly(rawContent);
//        return chatClient.prompt()
//                .user("你是一个面试官。请把下面这道死板的面试题改写成一句自然、口语化的面试开场提问，针对话题：" + topic + "。题目是：" + question)
//                .call().content();
//    }
//
//    private Document getQuestionFromVectorDB(String topic) {
//        List<Document> docs = vectorStore.similaritySearch(SearchRequest.query(topic).withTopK(5));
//        return docs.isEmpty() ? new Document("面试题：请聊聊" + topic + "。\n标准答案：略") : docs.get(new Random().nextInt(docs.size()));
//    }
//
//    private String extractQuestionOnly(String rawContent) {
//        if (rawContent.contains("标准答案")) {
//            return rawContent.split("标准答案")[0].replace("面试题：", "").trim();
//        }
//        return rawContent;
//    }

//    public ChatResult chat(String userText, String sessionId, String modelName) {
//
//        String constraint = """
//        # 禁令（必须严格遵守）：
//        - 严禁在回复中包含任何关于你“如何思考”、“如何改写”或“为什么这样问”的解释。
//        - 严禁出现“这个版本”、“评价如下”、“下一步追问”等引导词。
//        - 严禁使用括号提供备注。
//        - 你必须完全沉浸在面试官的角色中，你的输出就是你对面试者说的话。
//        """;
//
//        System.out.println("\n" + "█".repeat(60));
//        System.out.println("【SYS】Session: " + sessionId + " | 用户输入: " + userText);
//
//        InterviewSession session = sessionStates.computeIfAbsent(sessionId, k -> new InterviewSession());
//
//        // --- 阶段 0: 结束判定 ---
//        if (session.isFinished()) {
//            return new ChatResult(userText, "本次面试已经圆满结束，期待我们的下次见面。");
//        }
//
//        // --- 阶段 1: START 握手（开场白） ---
//        if ("START".equalsIgnoreCase(userText)) {
//            session.setStep(1);
//            return new ChatResult(userText, "你好！我是今天的技术面试官。请先做一个简单的自我介绍，并告诉我你想应聘的岗位是？");
//        }
//
//        // --- 阶段 2: 话题锁定与首题抛出 ---
//        if (session.getStep() == 1) {
//            // 1. 语义纠错层：利用 LLM 修复 STT 的错误
//            String correctedText = chatClient.prompt()
//                    .user("你是面试官的语音识别纠错助手。我们刚才让面试者进行自我介绍，以及询问他想应聘的岗位是什么，请下面段可能有误的受试者的回答进行修正，只需返回修正后的文字。" +
//                            "。原始输入为：" + userText)
//                    .options(OllamaOptions.builder().withModel(modelName).build())
//                    .call().content().trim();
//
//            System.out.println("【DEBUG】STT 原文: " + userText);
//            System.out.println("【DEBUG】LLM 纠错后: " + correctedText);
//
//            userText = correctedText;
//
//            System.out.println("【STEP 1】正在分析用户背景并锁定 3 个核心 Topic...");
//
//            List<String> allTopics = graphService.getAllTopics();
//            String selectPrompt = String.format("用户介绍：'%s'。从列表%s中选3个最相关的技术话题名。只返回名称并用逗号分隔。",
//                    userText, allTopics);
//            System.out.println("锁定主题使用的Prompt:"+ selectPrompt);
//            String rawTopics = chatClient.prompt().user(selectPrompt).options(OllamaOptions.builder().withModel(modelName).build()).call().content();
//            List<String> locked = Arrays.asList(rawTopics.split(",")).stream()
//                    .map(String::trim).collect(Collectors.toList());
//
//            session.setLockedTopics(locked);
//            session.setCurrentTopicIndex(0);
//            session.setStep(2);
//
//            System.out.println("【DEBUG】已锁定的面试路径: " + locked);
//
//            // 获取第一个话题的首题
//            String firstTopic = locked.getFirst();
//            Document firstDoc = getQuestionFromVectorDB(firstTopic);
//            session.setLastExpectedAnswer(firstDoc.getContent()); // 预存答案给下一轮评价
//
//            String firstQuestion = polishQuestion(firstTopic, firstDoc.getContent(), modelName);
//            System.out.println("【DEBUG】首题预存答案: " + firstDoc.getContent());
//
//            session.setLastQuestion(firstQuestion);
//            return  new ChatResult(correctedText, firstQuestion);
//        }
//
//        // --- 阶段 3: 循环面试（实体锚定 + 图谱跳转 + 向量搜题） ---
//        System.out.println("【STEP 2】执行深度 RAG 循环逻辑...");
//
//        // 1. 获取当前主话题
//        String currentMainTopic = session.getLockedTopics().get(session.getCurrentTopicIndex());
//
//        // --- 【核心改动 1：语义纠错】 ---
//        userText = correctSpeechText(userText, session.getLastQuestion(), modelName);
//
//
//        // 2. [实体锚定] 让 LLM 从用户回答中提取当前讨论的核心实体
//        // 为了提高准确率，我们把当前 Topic 下的所有图谱 Head 传给它作为参考
//        String extractPrompt = String.format("""
//        当前面试主题是：%s。
//        用户回答了：'%s'。
//        请从用户回答中提取一个核心技术名词（实体）。
//        必须从这个名单中挑选最接近的：%s。
//        只需返回名词，不要解释。
//        """, currentMainTopic, userText, graphService.getAllHeadsByTopic(currentMainTopic));
//
//        String anchorEntity = chatClient.prompt().user(extractPrompt).options(OllamaOptions.builder().withModel(modelName).build()).call().content().trim();
//        System.out.println("【DEBUG】LLM 锚定到的实体: " + anchorEntity);
//
//        // 3. [图谱跳转] 在知识图谱中寻找该实体的“下一跳”
//        Triplet nextFact = graphService.findNextStep(anchorEntity, currentMainTopic);
//        String nextSearchKey = currentMainTopic; // 默认搜索词
//        String graphLogicHint = "继续深入探讨。";
//
//        if (nextFact != null) {
//            nextSearchKey = nextFact.getTail(); // 跳转到尾实体，例如从“分布式通知”跳转到“ZooKeeper节点状态变化”
//            graphLogicHint = String.format("注意到用户提到了%s，其%s是%s，我们可以据此深入。",
//                    nextFact.getHead(), nextFact.getRelation(), nextFact.getTail());
//            System.out.println("【DEBUG】图谱路径跳转成功: " + nextFact);
//        } else {
//            System.out.println("【DEBUG】图谱未发现直接路径，保持当前主题检索。");
//        }
//
//        // 4. [向量检索] 使用跳转后的新实体去向量库搜题
//        Document nextDoc = getQuestionFromVectorDB(nextSearchKey);
//        final String lastRefAnswer = session.getLastExpectedAnswer(); // 上轮存好的标准答案
//        session.setLastExpectedAnswer(nextDoc.getContent()); // 存入本轮搜到的答案，供下轮评价用
//
//        System.out.println("【DEBUG】本轮评价参考（上轮答案）: " + lastRefAnswer);
//        System.out.println("【DEBUG】本轮搜到的新题（标准答案）: " + nextDoc.getContent());
//
//        // 5. [状态切换判定]
//        session.setQuestionCount(session.getQuestionCount() + 1);
//        boolean shouldSwitch = session.getQuestionCount() >= 3;
//
//        String systemPrompt;
//        final String nextRawQuestion = extractQuestionOnly(nextDoc.getContent());
//        final String finalGraphFacts = graphLogicHint;
//
//        if (shouldSwitch && session.getCurrentTopicIndex() >= session.getLockedTopics().size() - 1) {
//            // 面试总收尾
//            session.setFinished(true);
//            System.out.println("【DEBUG】判定：所有话题结束，进入总评。");
//            systemPrompt = """
//            你是一位资深面试官。
//            1. 评价：参考[标准答案]对用户刚才的回答做简短点评。
//            2. 总结：对用户今天的整体表现做一个专业且有温度的总结。
//            3. 结束：礼貌地结束面试。
//
//            """ + constraint + """
//
//            [标准答案]: {expectedAnswer}
//            """;
//        } else if (shouldSwitch) {
//            // 切换到下一个主 Topic
//            session.setCurrentTopicIndex(session.getCurrentTopicIndex() + 1);
//            session.setQuestionCount(0);
//            String newTopic = session.getLockedTopics().get(session.getCurrentTopicIndex());
//            System.out.println("【DEBUG】判定：切换主话题 -> " + newTopic);
//            systemPrompt = """
//            你是一位面试官。
//            1. 评价：参考[标准答案]评价用户回答。
//            2. 转换：告诉用户关于上一个话题聊得不错，现在我们转向下一个领域：""" + newTopic + """
//            3. 提问：请把[原始题干]改写成自然的面试发问：{nextRawQuestion}。
//
//            """ + constraint + """
//
//            [标准答案]: {expectedAnswer}
//            """;
//        } else {
//            // 同一 Topic 内顺着图谱继续问
//            System.out.println("【DEBUG】判定：顺着图谱逻辑继续追问。");
//            systemPrompt = """
//            你是一位资深面试官。请组织一段流利、没有人情味的、自然的对话：
//            1. 评价：参考[标准答案]简短评价用户回答（如：答到了点子上、理解有误等）。
//            2. 衔接：利用[图谱逻辑提示]中的关系进行过渡。
//            3. 追问：将[原始题干]改写成口语化的追问：{nextRawQuestion}。
//
//            """ + constraint + """
//
//            [标准答案]: {expectedAnswer}
//            [图谱逻辑提示]: {graphFacts}
//            """;
//        }
//
//        // 6. 最终合成回复
//        String response = chatClient.prompt()
//                .system(s -> s.text(systemPrompt)
//                        .param("expectedAnswer", lastRefAnswer)
//                        .param("graphFacts", finalGraphFacts)
//                        .param("nextRawQuestion", nextRawQuestion))
//                .user(userText)
//                .options(OllamaOptions.builder().withModel(modelName).build())
//                .advisors(a -> a.param(CHAT_MEMORY_CONVERSATION_ID_KEY, sessionId))
//                .call()
//                .content();
//
//        System.out.println("【DEBUG】面试官回复: " + response);
//        System.out.println("█".repeat(60));
//
//        session.setLastQuestion(response);
//        return new ChatResult(userText, response);
//    }

    private String generateFinalReport(String sessionId, String modelName) {
        System.out.println(">>> 正在生成最终面试报告...");

        // 这个 Prompt 不需要 userText，直接让模型回顾历史
        String summaryPrompt = """
        面试已经结束。请你作为首席面试官，根据刚才所有的对话历史记录，对面试者的表现进行综合评价。
        
        请严格按以下格式输出：
        好的，今天的面试就到这里。
        【面试评分】：分数/100
        【面试评价】：针对面试者的技术广度、深度及表达能力进行总结，指出其亮点和需要加强的地方。
        【最终结果】：通过/不通过（评分60以上为通过）
        
        要求：
        1. 语气专业且客观。
        2. 评价要基于刚才实际聊过的技术点（如ROS、Java、C++等）。
        3. 不要输出除上述格式以外的其他任何内容。
        """;

        try {
            return chatClient.prompt()
                    .user(summaryPrompt)
                    .options(OllamaOptions.builder().withModel(modelName).build())
                    // 核心：通过 sessionId 让 Advisor 把刚才聊天的全过程历史塞给模型
                    .advisors(a -> a.param(CHAT_MEMORY_CONVERSATION_ID_KEY, sessionId))
                    .call()
                    .content();
        } catch (Exception e) {
            System.err.println("报告生成失败: " + e.getMessage());
            return "面试已结束，感谢参与。由于系统原因未能生成详细报告，请联系管理员。";
        }
    }

    public ChatResult chat(String userText, String sessionId, String modelName) {
        System.out.println("\n" + "*".repeat(60));
        InterviewSession session = sessionStates.computeIfAbsent(sessionId, k -> new InterviewSession());

        // 0. 结束判定
        if (session.isFinished() || session.getTotalValidKeywordsHandled() >= InterviewSession.KEYWORD_THRESHOLD) {
//            session.setFinished(true);
//            return new ChatResult(userText, "面试已结束，感谢。");
            // 如果是第一次进入结束状态
            if (!session.isFinished()) {
                session.setFinished(true); // 锁定状态
                String finalReport = generateFinalReport(sessionId, modelName);
                return new ChatResult(userText, finalReport);
            }

            // 如果已经结束过了，用户又发了消息
            return new ChatResult(userText, "面试已圆满结束，感谢您的参与。请刷新页面开启新会话。");
        }

        // 1. START 阶段
        if ("START".equalsIgnoreCase(userText)) {
            session.setStep(1);
            String welcome = "你好！我是面试官。请问你今天应聘的是什么岗位？可以简单介绍下你的技术栈吗？";
            session.setLastQuestion(welcome);
            return new ChatResult(userText, welcome);
        }

        // --- 统一纠错处理 ---
        userText = correctSpeechText(userText, session.getLastQuestion(), modelName);

        // 2. 岗位介绍阶段 -> 生成 List_A (大纲)
        if (session.getStep() == 1) {
            System.out.println("【STEP 1】识别岗位关键词...");
            List<String> allKGs = graphService.getAllEntities();
            String selectPrompt = String.format("""
            用户介绍了背景：'%s'。
            请从这些技术点中猜测面试可能涉及的5-8个核心概念名词。
            只返回名称，逗号分隔。
            """, userText, allKGs);

            String rawListA = chatClient.prompt().user(selectPrompt).options(OllamaOptions.builder().withModel(modelName).build()).call().content();
            session.setTechKeywordsListA(Arrays.asList(rawListA.split(",")).stream().map(String::trim).collect(Collectors.toList()));
            session.setStep(2);
            session.setKeywordIndex(0);

            System.out.println("【DEBUG】生成的大纲 List_A: " + session.getTechKeywordsListA());

            // 抛出基于 List_A 第一个词的题目
            return getNextQuestionByNewKeyword(userText, session, modelName, sessionId);
        }

        // 3. 循环面试阶段 (评价 + 图谱跳跃 + 搜题)
        System.out.println("【STEP 2】评价并寻找下一跳...");

        // A. 提取用户回答中的实体
        String currentEntity = extractEntityFromAnswer(userText, modelName);
        System.out.println("【DEBUG】提取到的回答实体: " + currentEntity);

        // B. 图谱查找邻居 (下一跳候选)
        List<String> neighbors = graphService.findNeighbors(currentEntity);
        System.out.println("【DEBUG】图谱找到的邻居: " + neighbors);

        Document nextDoc = null;
        // C. 尝试从邻居中找一个“没问过”的问题
        for (String neighbor : neighbors) {
            nextDoc = searchVectorDBUnique(neighbor, session);
            if (nextDoc != null) {
                System.out.println("【DEBUG】图谱跳跃成功，找到新题: " + neighbor);
                break;
            }
        }

        // D. 如果邻居都问过了或没邻居，则切换回 List_A 大纲
        if (nextDoc == null) {
            System.out.println("【DEBUG】图谱路径用尽，切换大纲关键词...");
            session.setKeywordIndex(session.getKeywordIndex() + 1);
            if (session.getKeywordIndex() >= session.getTechKeywordsListA().size()) {
                return handleInterviewEnd(userText, session, modelName);
            }
            return getNextQuestionByNewKeyword(userText, session, modelName, sessionId);
        }

        // E. 正常执行：评价 + 抛出新题
        return composeResponse(userText, nextDoc, session, modelName, sessionId);
    }

// --- 核心辅助工具函数 ---

    /**
     * 从向量库搜寻题目，并确保不重复
     */
//    private Document searchVectorDBUnique(String queryKey, InterviewSession session) {
//        List<Document> docs = vectorStore.similaritySearch(SearchRequest.query(queryKey).withTopK(5));
//        for (Document d : docs) {
//            // 假设 metadata 中存有 id 或者 topic 含有编号，比如 "题目 001"
//            String qId = (String) d.getMetadata().getOrDefault("topic", d.getContent().substring(0, 10));
//            if (!session.getUsedQuestionIds().contains(qId)) {
//                session.getUsedQuestionIds().add(qId);
//                return d;
//            }
//        }
//        return null;
//    }

    private Document searchVectorDBUnique(String queryKey, InterviewSession session) {
        System.out.println("\n" + "·".repeat(20) + " [向量库检索开始] " + "·".repeat(20));
        System.out.println("【DEBUG-VEC】检索关键词 (queryKey): [" + queryKey + "]");

        // 1. 执行向量相似度搜索
        List<Document> docs = vectorStore.similaritySearch(SearchRequest.query(queryKey).withTopK(5));

        if (docs == null || docs.isEmpty()) {
            System.out.println("【DEBUG-VEC】结果：未在向量库中找到任何相关内容。");
            return null;
        }

        System.out.println("【DEBUG-VEC】检索到 " + docs.size() + " 条候选文档：");
//        System.out.println(docs);

        for (int i = 0; i < docs.size(); i++) {
            Document d = docs.get(i);
            Map<String, Object> metadata = d.getMetadata();

            // 2. 获取判重 ID（如果 topic 为空，取内容前15个字，防止前面几道题开头都一样导致碰撞）
            String topicMeta = (String) metadata.get("topic");
            String qId = (topicMeta != null && !topicMeta.isEmpty())
                    ? topicMeta
                    : d.getContent().substring(0, Math.min(d.getContent().length(), 15)).trim();

            // 3. 准备打印内容预览（取前 50 字并去掉换行）
            String contentPreview = d.getContent().substring(0, Math.min(d.getContent().length(), 50)).replace("\n", " ");

            System.out.println(String.format("  ➤ 候选 [%d]:", i));
            System.out.println("     - [ID/Topic]: " + qId);
            System.out.println("     - [Metadata]: " + metadata);
            System.out.println("     - [内容预览]: " + contentPreview + "...");

            // 4. 执行判重逻辑
            if (!session.getUsedQuestionIds().contains(qId)) {
                System.out.println("     ✅ 决策：未曾问过，选定此题。");
                session.getUsedQuestionIds().add(qId);
                System.out.println("·".repeat(50) + "\n");
                return d;
            } else {
                System.out.println("     ❌ 决策：已在 UsedQuestionIds 列表中，跳过。");
            }
        }

        System.out.println("【DEBUG-VEC】结果：所有候选项都已问过。");
        System.out.println("·".repeat(50) + "\n");
        return null;
    }

    /**
     * 当切入一个全新的大纲关键词时执行
     */
    private ChatResult getNextQuestionByNewKeyword(String userText, InterviewSession session, String modelName, String sessionId) {
        String currentKeyword = session.getTechKeywordsListA().get(session.getKeywordIndex());
        Document doc = searchVectorDBUnique(currentKeyword, session);

        if (doc == null) { // 如果大纲里的词也没题，递归找下一个大纲词
            session.setKeywordIndex(session.getKeywordIndex() + 1);
            if (session.getKeywordIndex() >= session.getTechKeywordsListA().size())
                return handleInterviewEnd(userText, session, modelName);
            return getNextQuestionByNewKeyword(userText, session, modelName, sessionId);
        }

        session.setTotalValidKeywordsHandled(session.getTotalValidKeywordsHandled() + 1);
        return composeResponse(userText, doc, session, modelName, sessionId);
    }

    /**
     * 从用户回答中提取核心技术实体
     */
    private String extractEntityFromAnswer(String userText, String modelName) {
        // 1. 获取图谱中所有已知的实体列表（可选，作为参考给模型，能极大提高匹配率）
        // 如果实体列表太大（过万），则不建议全量传入，可以只传一个简单的提取指令
        List<String> allKnownEntities = graphService.getAllEntities();

        // 我们只取前 200 个或者不传，防止 Prompt 过长，这里采用高度约束的指令
        String prompt = String.format("""
            你是一个技术名词提取专家。
            任务：从面试者的回答中提取出一个【最核心】的技术名词（实体）。
            
            面试者回答内容：'%s'
            
            要求：
            1. 必须是计算机专业术语（如：分布式锁、JVM、高并发、一致性哈希等）。
            2. 只能返回名词本身，严禁包含任何解释、标点或括号。
            3. 如果没有发现明显的技术术语，请返回'None'。
            """, userText);

        try {
            String entity = chatClient.prompt()
                    .user(prompt)
                    .options(OllamaOptions.builder().withModel(modelName).build())
                    .call()
                    .content()
                    .trim();

            // 基础清洗：防止大模型固执地吐出“实体：ZooKeeper”或者带有句号
            entity = entity.replace("实体：", "").replace("实体:", "")
                    .replace("。", "").replace(".", "").trim();

            return entity;
        } catch (Exception e) {
            System.err.println(">>> [ERROR] 实体提取环节异常: " + e.getMessage());
            return "None";
        }
    }

    /**
     * 组装：评价上轮 + 抛出下轮
     */
    private ChatResult composeResponse(String userText, Document nextDoc, InterviewSession session, String modelName, String sessionId) {
        System.out.println("尝试回复中~");
        final String lastAnswer = session.getLastExpectedAnswer(); // 此时是针对用户当前回答的标准解
        final String nextFullContent = nextDoc.getContent();
        final String nextQ = extractQuestionOnly(nextFullContent);
        session.setLastExpectedAnswer(nextFullContent); // 更新为下一轮做准备

        // 核心改进 1: 使用 {expectedAnswer} 占位符代替 %s
        // 核心改进 2: 删除了末尾的 .formatted(...)
        String promptTemplate = """
        你是一位资深面试官。
        任务：
        1. 参考[上题标准答案]对用户刚才的回答简短评价。如果上题没有参考答案，说明是第一次提问，你需要表现得像首次发问一样。如果面试者没有回答出来，那么适当安慰他，
        2. 衔接并提出新问题：{nextQ}。要求改写得像真人说话。
        
        [上题标准答案]: {expectedAnswer}
        
        ### 极其重要的约束（违者面试失败）：
        1. 必须保持专业且口语化。
        2. 你所输出的，就是面试官对受试者说的话，不要输出无关信息
        3. 语气温和
        """;

        String response = chatClient.prompt()
                .system(s -> s.text(promptTemplate)
                        .param("nextQ", nextQ)
                        // 核心改进 3: 将答案内容通过 param 传入，ST4 引擎会安全处理其中的代码/花括号
                        .param("expectedAnswer", lastAnswer.isEmpty() ? "这是第一题，无需评价" : lastAnswer))
                .user(userText)
                .options(OllamaOptions.builder().withModel(modelName).build())
                // 确保带上 sessionId 保持多轮记忆
                .advisors(a -> a.param(CHAT_MEMORY_CONVERSATION_ID_KEY, sessionId))
                .call()
                .content();

        session.setLastQuestion(response);
        return new ChatResult(userText, response);
    }

    /**
     * 结束流程
     */
    private ChatResult handleInterviewEnd(String userText, InterviewSession session, String modelName) {
        session.setFinished(true);
        String summary = chatClient.prompt()
                .user("面试结束。请根据历史表现对用户进行综合评分和优缺点总结。")
                .options(OllamaOptions.builder().withModel(modelName).build())
                .call().content();
        return new ChatResult(userText, "好的，今天的技术考察到此为止。总结如下：\n" + summary);
    }


    // --- 辅助方法 ---
    private String polishQuestion(String topic, String rawContent, String modelName) {
        String q = extractQuestionOnly(rawContent);
        return chatClient.prompt()
                .user(String.format("""
                你是一个资深技术面试官。
                任务：将以下死板的题目改写成一句自然的、真人在面试现场会问出的口语化提问。
                
                话题：[%s]
                原始题目：[%s]
                
                ### 极其重要的约束（违者面试失败）：
                1. 必须保持专业且口语化。
                2. **只返回改写后的那一句话内容**。
                3. **严禁包含任何括号、解释、评价、改进说明或“这个版本通过...方式”等字样**。
                4. **禁止输出任何除题目本身以外的文字**。
                """, topic, q))
                .options(OllamaOptions.builder().withModel(modelName).build())
                .call().content().trim();
    }

    private Document getQuestionFromVectorDB(String topic) {
        List<Document> docs = vectorStore.similaritySearch(SearchRequest.query(topic).withTopK(5));
        return docs.isEmpty() ? new Document("面试题：请聊聊" + topic + "。\n标准答案：略") : docs.get(new Random().nextInt(docs.size()));
    }

    private String extractQuestionOnly(String rawContent) {
        if (rawContent.contains("标准答案")) {
            return rawContent.split("标准答案")[0].replace("面试题：", "").trim();
        }
        return rawContent;
    }

    private String correctSpeechText(String rawText, String lastQuestion, String modelName) {
        // 如果没有上一个问题（比如自我介绍阶段），直接返回原样
        if (lastQuestion == null || lastQuestion.isEmpty()) {
            return rawText;
        }

        try {
            String correctionPrompt = String.format("""
            你是一个计算机技术专家。现在正在辅佐面试官对被面试者进行面试，由于语音识别(STT)在处理专业词汇时可能出错，请你根据【上一个面试问题】来修复面试者回答的【原始识别文本】中的技术术语错误。
            
            【上一个面试问题】：%s
            【原始识别文本】：%s
            
            要求：
            1. 仅修复技术名词（如：把"猪Keeper"修复为"ZooKeeper"，把"JBM"修复为"JVM"）。
            2. 保持原有的句式和语气。
            3. 如果识别文本基本正确，请原样返回。
            4. **只返回修复后的最终文本，严禁任何解释。**
            """, lastQuestion, rawText);

            String corrected = chatClient.prompt()
                    .user(correctionPrompt)
                    .options(OllamaOptions.builder().withModel(modelName).build())
                    .call().content().trim();

            System.out.println("【DEBUG-STT】修正前: " + rawText);
            System.out.println("【DEBUG-STT】修正后: " + corrected);
            return corrected;
        } catch (Exception e) {
            System.err.println(">>> 语义纠错失败: " + e.getMessage());
            return rawText; // 失败则容错，使用原文本
        }
    }

    public String chat_2(String userText, String sessionId, String modelName) {
        System.out.println("\n" + "⚡".repeat(60));
        System.out.println("【chat_2】Session: " + sessionId + " | 原始输入: " + userText);


        // 👈 核心修复点：定义一个 final 变量供 Lambda 使用
        final String finalRagContext = userText;

        // 2. 构建 System Prompt
        String systemPrompt = """
        你是一位专业的计算机专业面试官，风格严谨，喜欢追问底层原理。请根据候选人的回答进行追问或点评。面试中对话不超过10轮，完成面试时面试官主动结束并给出打分和点评。
        """;

        // 3. 调用大模型
        try {
            return chatClient.prompt()
                    // 👈 这里使用 finalRagContext
                    .system(s -> s.text(systemPrompt).param("ragContext", finalRagContext))
                    .user(userText)
                    .options(OllamaOptions.builder().withModel(modelName).build())
                    .advisors(a -> a
                            .param(CHAT_MEMORY_CONVERSATION_ID_KEY, sessionId)
                            .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 15))
                    .call()
                    .content();
        } catch (Exception e) {
            e.printStackTrace();
            return "面试官信号灯闪烁，请稍后再试: " + e.getMessage();
        }
    }

    @Data
    @AllArgsConstructor
    public static class ChatResult {
        private String correctedUserText; // 纠错后的用户说话内容
        private String aiResponse;        // AI 的回答
    }
}