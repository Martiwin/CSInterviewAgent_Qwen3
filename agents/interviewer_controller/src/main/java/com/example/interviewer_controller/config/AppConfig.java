package com.example.interviewer_controller.config;

import org.springframework.ai.chroma.ChromaApi;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.ChromaVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier; // 👈 导入这个
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AppConfig {

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    @Bean
    @Primary
    public VectorStore vectorStore(@Qualifier("openAiEmbeddingModel") EmbeddingModel embeddingModel) {
        // 这里的 @Qualifier("openAiEmbeddingModel") 强制选择了 SiliconFlow 提供的模型

        ChromaApi chromaApi = new ChromaApi("http://localhost:8001");

        // 按照之前的报错，参数顺序为：EmbeddingModel, ChromaApi, String, boolean
        return new ChromaVectorStore(embeddingModel, chromaApi, "langchain", false);
    }
}