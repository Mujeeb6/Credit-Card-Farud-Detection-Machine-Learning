# Production ML Fraud Detection System 💳
[![CI Pipeline - Docker Build Test](https://github.com/Mujeeb6/Credit-Card-Farud-Detection-Machine-Learning/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Mujeeb6/Credit-Card-Farud-Detection-Machine-Learning/actions)

A production-ready machine learning system that detects fraudulent credit card transactions. This project demonstrates core MLOps principles by serving Scikit-Learn and XGBoost models through a containerized, monitored web application.

## 🚀 Overview
This application bridges the gap between data science research and scalable production. It provides a robust pipeline that handles imbalanced data preprocessing, model training, and evaluation, all served via a RESTful Flask application running on a production-grade Gunicorn server.

## 🛠️ Tech Stack
* **MLOps & Infrastructure:** Docker, GitHub Actions (CI/CD), Python `logging`
* **Application Framework:** Python, Flask, Gunicorn
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, SMOTE (imbalanced-learn)

## ⚙️ System Workflow
1. **Automated Preprocessing:** Handles missing data, applies `StandardScaler` for feature scaling, and balances highly skewed datasets using SMOTE to ensure reliable model training.
2. **Model Serving:** Provides a dynamic interface to train and evaluate Logistic Regression, Decision Trees, and XGBoost models on the fly.
3. **Containerization:** Encapsulated in a lightweight Docker container for consistent cross-environment deployment.
4. **Monitoring & Logging:** Implements structured Python logging to track system startup, inference requests, and catch runtime errors.
5. **CI/CD:** GitHub Actions automatically tests the Docker image build process on every push to the `main` branch.

## 💻 Running Locally (Docker)

**1. Clone the repository:**
```bash
git clone [https://github.com/Mujeeb6/Credit-Card-Farud-Detection-Machine-Learning.git](https://github.com/Mujeeb6/Credit-Card-Farud-Detection-Machine-Learning.git)
cd Credit-Card-Farud-Detection-Machine-Learning