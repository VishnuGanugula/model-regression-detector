package com.example.modelregression.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ClassificationRequest {

    @JsonProperty("email_text")
    private String emailText;

    public ClassificationRequest() {
    }

    public ClassificationRequest(String emailText) {
        this.emailText = emailText;
    }

    public String getEmailText() {
        return emailText;
    }

    public void setEmailText(String emailText) {
        this.emailText = emailText;
    }
}
