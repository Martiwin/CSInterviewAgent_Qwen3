package com.example.interviewer_controller.service;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Data
@NoArgsConstructor
class InterviewSession {
    private List<String> techKeywordsListA = new ArrayList<>(); // 岗位关键词大纲 (List_A)
    private int keywordIndex = 0;                               // 当前在大纲中的位置
    private Set<String> usedQuestionIds = new HashSet<>();      // 记录已问过的问题编号
    private String lastExpectedAnswer = "";                     // 上一题标准答案
    private String lastQuestion = "";                           // 上一个提出的问题
    private int totalValidKeywordsHandled = 0;                  // 已有效匹配的关键词数量
    private int step = 0;
    private boolean isFinished = false;

    // 阈值常量
    public static final int KEYWORD_THRESHOLD = 2;              // 问满5个关键词大点就结束
}
