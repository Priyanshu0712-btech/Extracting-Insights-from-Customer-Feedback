-- Minimal schema for the app
CREATE DATABASE IF NOT EXISTS customer_feedback_db;
USE customer_feedback_db;

CREATE TABLE IF NOT EXISTS users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(150) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS reviews (
  review_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  review_text TEXT,
  product_id VARCHAR(100),
  category VARCHAR(100),
  uploaded_at DATETIME,
  overall_sentiment VARCHAR(50),
  overall_sentiment_score FLOAT,
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
