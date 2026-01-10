package com.example.interviewer_controller.controller;

import com.example.interviewer_controller.service.InterviewService;
import com.example.interviewer_controller.service.SpeechService;
import org.springframework.web.bind.annotation.*;
        import org.springframework.web.multipart.MultipartFile;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class InterviewController {

    private final SpeechService speechService;
    private final InterviewService interviewService;

    // Spring 会自动把上面写好的两个 Service 注入进来
    public InterviewController(SpeechService speechService, InterviewService interviewService) {
        this.speechService = speechService;
        this.interviewService = interviewService;
    }

    @PostMapping("/interview")
    public Map<String, Object> handleInterview(
            // 关键：required = false，允许不传文件（第一次初始化只有文字）
            @RequestParam(value = "file", required = false) MultipartFile file,
            // 关键：增加 text 参数，用于接收 "START" 信号
            @RequestParam(value = "text", required = false) String text,
            @RequestParam(value = "sessionId", defaultValue = "test-session") String sessionId,
            @RequestParam(value = "mode", defaultValue = "finetuned") String mode) {


        // 加上这个日志，如果你能在控制台看到这行，说明请求进来了！
        System.out.println(">>> 接口收到请求: text=" + text + ", hasFile=" + (file != null));

        String userText = "";
        if (file != null && !file.isEmpty()) {
            userText = speechService.speechToText(file.getResource());
        } else {
            userText = text; // 如果没语音，就用文字（处理 START）
        }

        // 逻辑分发：如果 userText 是空的，给个默认
        if (userText == null || userText.isEmpty()) {
            userText = "START";
        }

        String aiResponse;
        System.out.println("mode" + mode);
        if ("base".equalsIgnoreCase(mode)) {
            // 模式一：原始模型 + 复杂逻辑 + 知识图谱
            System.out.println(">>> [切换至：基础模式]");
            InterviewService.ChatResult resultObj = interviewService.chat(userText, sessionId, "qwen3:8b"); // 假设基础模型名是这个
            // 3. 组装返回给前端
            Map<String, Object> result = new HashMap<>();
            // 关键：这里传回的是 resultObj 里的 correctedUserText (纠错后的)
            userText = resultObj.getCorrectedUserText();
            aiResponse = resultObj.getAiResponse();
        } else {
            // 模式二：微调模型 + 简化逻辑
            System.out.println(">>> [切换至：微调模式]");
            aiResponse = interviewService.chat_2(userText, sessionId, "interviewer-qwen3");
        }

        byte[] audioBytes = speechService.textToSpeech(aiResponse);

        Map<String, Object> result = new HashMap<>();
        result.put("userText", userText);
        result.put("aiText", aiResponse);
        result.put("audio", Base64.getEncoder().encodeToString(audioBytes));
        return result;
    }
}