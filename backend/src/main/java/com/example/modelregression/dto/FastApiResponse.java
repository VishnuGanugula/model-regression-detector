package com.example.modelregression.dto;

public class FastApiResponse {

    private String category;

    public FastApiResponse() {
    }

    public FastApiResponse(String category) {
        this.category = category;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }
}
