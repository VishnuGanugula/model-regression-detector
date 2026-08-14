-- MySQL Database Initialization Script

CREATE DATABASE IF NOT EXISTS email_classifier_db;
USE email_classifier_db;

CREATE TABLE IF NOT EXISTS classification_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email_text TEXT NOT NULL,
    predicted_category VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
