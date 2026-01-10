package com.example.interviewer_controller;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
// 导入这个需要排除的类
import org.springframework.ai.autoconfigure.vectorstore.chroma.ChromaVectorStoreAutoConfiguration;

@SpringBootApplication(exclude = {
		org.springframework.ai.autoconfigure.vectorstore.chroma.ChromaVectorStoreAutoConfiguration.class
})
public class InterviewerControllerApplication {
	public static void main(String[] args) {
		SpringApplication.run(InterviewerControllerApplication.class, args);
	}
}