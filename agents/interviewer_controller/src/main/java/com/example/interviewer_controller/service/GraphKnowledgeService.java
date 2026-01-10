package com.example.interviewer_controller.service;

import com.example.interviewer_controller.model.Triplet;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class GraphKnowledgeService {

    // 存储所有的原始三元组
    private List<Triplet> allTriplets;

    // 缓存所有去重后的实体名词（包括 Head 和 Tail）
    private List<String> allEntities;

    @PostConstruct
    public void init() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        // 加载知识图谱 JSON 文件
        this.allTriplets = mapper.readValue(
                new File("E:/Code/LLM_Base_and_Application/RAG/knowledge_graph.json"),
                new TypeReference<List<Triplet>>(){}
        );

        // 提取所有不重复的实体（Head 和 Tail 都要，因为任何名词都可能被面试者提到）
        Set<String> entitySet = new HashSet<>();
        for (Triplet t : allTriplets) {
            if (t.getHead() != null) entitySet.add(t.getHead().trim());
            if (t.getTail() != null) entitySet.add(t.getTail().trim());
        }
        this.allEntities = new ArrayList<>(entitySet);

        System.out.println("【图谱初始化完成】加载了 " + allTriplets.size() + " 条关系，共计 " + allEntities.size() + " 个核心实体。");
    }

    /**
     * 获取知识图谱中所有的核心技术实体名
     * 对应你设计的：让大模型从这个 List 中选择 List_A
     */
    public List<String> getAllEntities() {
        return this.allEntities;
    }

    /**
     * 核心逻辑：寻找实体的“邻居”
     * 当面试者提到 A 时，去图谱里找所有与 A 相关的 B、C、D...
     * 这些邻居将作为后续向量库搜索的凭证
     */
    public List<String> findNeighbors(String entity) {
        if (entity == null || entity.isEmpty()) return new ArrayList<>();

        String searchKey = entity.toLowerCase().trim();

        return allTriplets.stream()
                .filter(t -> t.getHead().toLowerCase().contains(searchKey) ||
                        t.getTail().toLowerCase().contains(searchKey))
                .map(t -> {
                    // 如果匹配到 head，返回 tail；反之亦然（实现双向寻找）
                    if (t.getHead().toLowerCase().contains(searchKey)) return t.getTail();
                    return t.getHead();
                })
                .distinct()
                .collect(Collectors.toList());
    }

    /**
     * 根据实体名获取关联的三元组事实
     * 用于给大模型提供“逻辑凭证”，解释为什么要问下一个题
     */
    public String getFactsByEntity(String entity) {
        String searchKey = entity.toLowerCase().trim();
        return allTriplets.stream()
                .filter(t -> t.getHead().toLowerCase().contains(searchKey) ||
                        t.getTail().toLowerCase().contains(searchKey))
                .limit(5)
                .map(t -> String.format("[%s] --(%s)--> [%s]", t.getHead(), t.getRelation(), t.getTail()))
                .collect(Collectors.joining("\n"));
    }

    /**
     * (保留旧接口名以防报错) 获取所有的话题（即三元组中的 source_topic）
     */
    public List<String> getAllTopics() {
        return allTriplets.stream()
                .map(Triplet::getSource_topic)
                .distinct()
                .collect(Collectors.toList());
    }
}