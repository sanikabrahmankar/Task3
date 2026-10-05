# Elevate Labs AI & ML Internship - Task 3: Linear Regression

## Overview
This repository contains the completion of Task 3 for the Elevate Labs AI & ML Internship. The objective was to implement and evaluate a simple linear regression model to predict housing prices based on area, utilizing Python and Scikit-Learn[cite: 3].

## Tools Used
Pandas: Data ingestion and preprocessing[cite: 3].
Matplotlib: Visualizing the regression line against actual data points[cite: 3].
Scikit-Learn: Splitting the dataset, training the Linear Regression model, and calculating evaluation metrics (MAE, MSE, $R^{2}$)[cite: 3].

## Steps Completed
1. Data Preprocessing: Loaded the `Housing.csv` dataset and isolated the predictor variable (`area`) and the target variable (`price`)[cite: 3].
2. Train-Test Split: Divided the data using an 80/20 split to ensure the model could be evaluated on unseen data[cite: 3].
3. Model Training: Fitted a Simple Linear Regression model using `sklearn.linear_model`[cite: 3].
4. Evaluation: Calculated Mean Absolute Error (MAE), Mean Squared Error (MSE), and the $R^{2}$ score to quantify model accuracy[cite: 3].
5. Interpretation: Extracted the slope (coefficient) and intercept to define the mathematical equation of the model[cite: 3].