package com.example.modelregression.entity;

import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "classification_history")
public class ClassificationHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "email_text", nullable = false, columnDefinition = "TEXT")
    private String emailText;

    @Column(name = "predicted_category", nullable = false, length = 50)
    private String predictedCategory;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    public ClassificationHistory() {
    }

    public ClassificationHistory(Long id, String emailText, String predictedCategory, LocalDateTime createdAt) {
        this.id = id;
        this.emailText = emailText;
        this.predictedCategory = predictedCategory;
        this.createdAt = createdAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getEmailText() {
        return emailText;
    }

    public void setEmailText(String emailText) {
        this.emailText = emailText;
    }

    public String getPredictedCategory() {
        return predictedCategory;
    }

    public void setPredictedCategory(String predictedCategory) {
        this.predictedCategory = predictedCategory;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public static ClassificationHistoryBuilder builder() {
        return new ClassificationHistoryBuilder();
    }

    public static class ClassificationHistoryBuilder {
        private Long id;
        private String emailText;
        private String predictedCategory;
        private LocalDateTime createdAt;

        public ClassificationHistoryBuilder id(Long id) {
            this.id = id;
            return this;
        }

        public ClassificationHistoryBuilder emailText(String emailText) {
            this.emailText = emailText;
            return this;
        }

        public ClassificationHistoryBuilder predictedCategory(String predictedCategory) {
            this.predictedCategory = predictedCategory;
            return this;
        }

        public ClassificationHistoryBuilder createdAt(LocalDateTime createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public ClassificationHistory build() {
            return new ClassificationHistory(id, emailText, predictedCategory, createdAt);
        }
    }
}
