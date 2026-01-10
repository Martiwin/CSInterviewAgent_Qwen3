package com.example.interviewer_controller.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Slf4j
@Service
public class SpeechService {

    private final RestTemplate restTemplate;

    @Value("${engines.stt.url}")
    private String sttUrl;

    @Value("${engines.tts.url}")
    private String ttsUrl;

    public SpeechService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * 调用 Faster-Whisper 将音频转为文字
     */
    public String speechToText(Resource audioResource) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", audioResource);
            body.add("model", "base");

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // 调用之前测试过的 8000 端口接口
            Map response = restTemplate.postForObject(sttUrl, requestEntity, Map.class);
            return response != null ? (String) response.get("text") : "";
        } catch (Exception e) {
            log.error("STT 转换失败", e);
            return "（语音识别失败）";
        }
    }

    /**
     * 调用 Edge-TTS 桥接服务将文字转为音频字节
     */
    public byte[] textToSpeech(String text) {
        try {
            // 调用之前测试过的 5000 端口接口
            return restTemplate.getForObject(ttsUrl + "?text=" + text, byte[].class);
        } catch (Exception e) {
            log.error("TTS 转换失败", e);
            return new byte[0];
        }
    }
}