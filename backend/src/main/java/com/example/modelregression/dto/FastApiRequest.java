package com.example.modelregression.dto;

public class FastApiRequest {

    private String text;

    public FastApiRequest() {
    }

    public FastApiRequest(String text) {
        this.text = text;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public static FastApiRequestBuilder builder() {
        return new FastApiRequestBuilder();
    }

    public static class FastApiRequestBuilder {
        private String text;

        public FastApiRequestBuilder text(String text) {
            this.text = text;
            return this;
        }

        public FastApiRequest build() {
            return new FastApiRequest(text);
        }
    }
}
