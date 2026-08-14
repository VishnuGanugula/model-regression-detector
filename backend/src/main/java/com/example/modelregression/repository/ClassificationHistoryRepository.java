package com.example.modelregression.repository;

import com.example.modelregression.entity.ClassificationHistory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ClassificationHistoryRepository extends JpaRepository<ClassificationHistory, Long> {

    /**
     * Retrieves all classification records ordered by timestamp (newest first).
     */
    List<ClassificationHistory> findAllByOrderByCreatedAtDesc();
}
