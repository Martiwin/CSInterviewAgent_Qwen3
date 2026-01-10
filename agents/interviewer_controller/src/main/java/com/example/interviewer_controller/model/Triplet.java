package com.example.interviewer_controller.model;

import lombok.Data;

@Data
public class Triplet {
    private String head;
    private String relation;
    private String tail;
    private String source_topic;

    @Override
    public String toString() {
        return String.format("[%s] --(%s)--> [%s]", head, relation, tail);
    }
}
