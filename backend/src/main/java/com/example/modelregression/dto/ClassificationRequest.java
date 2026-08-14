package com.example.modelregression.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class ClassificationRequest {

    @JsonProperty("email_text")
    private String emailText;
}
