# Marketing Subscription Prediction

## Overview
The Marketing Subscription Prediction project aims to predict whether a customer will subscribe to a marketing offer based on their past behavior, demographics, and interaction history. This project utilizes machine learning techniques to analyze the data and provide predictions, helping marketing teams identify the most promising leads and improve conversion rates.

## Problem Statement
In a highly competitive market, businesses often spend vast resources on marketing campaigns. The challenge is identifying which customers are more likely to subscribe to marketing offers, allowing businesses to focus their efforts and resources effectively. This project addresses the problem by building a predictive model using historical customer data.

## Features
- Predict customer subscription likelihood
- Evaluate and compare multiple machine learning algorithms (Logistic Regression, Random Forest, XGBoost, etc.)
- Data preprocessing, including handling missing values and feature engineering
- Model performance evaluation using accuracy, precision, recall, and F1-score

## Dataset
The dataset used in this project includes customer information such as:
- **Age**: Customer's age
- **Job**: Type of job the customer has
- **Marital Status**: Marital status of the customer
- **Education Level**: Customer’s education level
- **Previous Interaction**: Whether the customer has interacted with a similar campaign before
- **Contact**: Type of contact (email, phone, etc.)
- **Days Since Last Contact**: Time since the customer was last contacted
- **Outcome**: Whether the customer subscribed to the offer (target variable)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ZERO-legion5/MarketingSubscriptionPrediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd marketing-subscription-prediction
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```
