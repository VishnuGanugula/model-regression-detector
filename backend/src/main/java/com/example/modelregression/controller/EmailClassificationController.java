package com.example.modelregression.controller;

import com.example.modelregression.dto.ClassificationRequest;
import com.example.modelregression.dto.FastApiRequest;
import com.example.modelregression.dto.FastApiResponse;
import com.example.modelregression.entity.ClassificationHistory;
import com.example.modelregression.repository.ClassificationHistoryRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@RestController
@RequestMapping("/api/emails")
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
public class EmailClassificationController {

    private final ClassificationHistoryRepository repository;
    private final RestTemplate restTemplate;

    @Value("${fastapi.service.url:http://localhost:8000/predict}")
    private String fastapiServiceUrl;

    public EmailClassificationController(ClassificationHistoryRepository repository, RestTemplate restTemplate) {
        this.repository = repository;
        this.restTemplate = restTemplate;
    }

    /**
     * POST /api/emails/classify
     * Proxies the customer email text to Python FastAPI microservice for AI inference,
     * persists prediction to MySQL database, and returns saved record.
     */
    @PostMapping("/classify")
    public ResponseEntity<?> classifyEmail(@RequestBody ClassificationRequest request) {
        if (request.getEmailText() == null || request.getEmailText().trim().isEmpty()) {
            return ResponseEntity.badRequest().body("email_text is required.");
        }

        try {
            // 1. Prepare request payload for Python FastAPI microservice
            FastApiRequest fastApiPayload = FastApiRequest.builder()
                    .text(request.getEmailText().trim())
                    .build();

            // 2. Synchronous HTTP POST request to FastAPI microservice
            ResponseEntity<FastApiResponse> fastApiResponse = restTemplate.postForEntity(
                    fastapiServiceUrl,
                    fastApiPayload,
                    FastApiResponse.class
            );

            String predictedCategory = "unknown";
            if (fastApiResponse.getBody() != null && fastApiResponse.getBody().getCategory() != null) {
                predictedCategory = fastApiResponse.getBody().getCategory().toLowerCase();
            }

            // 3. Save entity to MySQL database
            ClassificationHistory record = ClassificationHistory.builder()
                    .emailText(request.getEmailText().trim())
                    .predictedCategory(predictedCategory)
                    .build();

            ClassificationHistory savedRecord = repository.save(record);

            return ResponseEntity.ok(savedRecord);

        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error communicating with AI service or database: " + e.getMessage());
        }
    }

    /**
     * GET /api/emails/history
     * Returns list of all past classification records ordered by newest first.
     */
    @GetMapping("/history")
    public ResponseEntity<List<ClassificationHistory>> getHistory() {
        List<ClassificationHistory> history = repository.findAllByOrderByCreatedAtDesc();
        return ResponseEntity.ok(history);
    }
}
