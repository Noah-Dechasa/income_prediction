# Adult Income Prediction

## Project Overview

This project focuses on predicting whether an individual earns more than $50K a year based on various demographic features using machine learning models. The goal is to classify individuals into two categories: those earning more than $50K and those earning less. The dataset used is the **Adult Income Dataset**, available from the **UCI Machine Learning Repository**.

## Dataset

The dataset contains demographic information such as age, education, marital status, occupation, and hours worked per week. The target variable is the income category, where:
- **<=50K** indicates individuals who earn $50K or less annually.
- **>50K** indicates individuals who earn more than $50K annually.

### Key Features:
- **age**: Age of the individual
- **education**: Education level (e.g., Bachelors, Masters)
- **marital-status**: Marital status (e.g., Married, Divorced)
- **occupation**: Occupation (e.g., Tech-support, Exec-managerial)
- **hours-per-week**: Number of hours worked per week
- **capital-gain**: Capital gains
- **capital-loss**: Capital losses

## Project Goal

The goal of this project was to build and evaluate different machine learning models for predicting whether an individual earns more than $50K annually. I used multiple models to understand which approach yields the best prediction accuracy.

### Models Used:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

### Key Steps:
1. **Data Preprocessing**: Cleaned the data, handled missing values, encoded categorical variables, and scaled numerical features.
2. **Model Training**: Trained multiple classification models to predict income levels based on the demographic features.
3. **Model Evaluation**: Evaluated the models using accuracy, precision, recall, and F1-score. A comparison of all models was made to find the best performer.

## Results

### XGBoost Model:
- **Accuracy**: 87.39%
- **Precision (Class 0)**: 0.90, **Recall (Class 0)**: 0.94
- **Precision (Class 1)**: 0.78, **Recall (Class 1)**: 0.67
- **Best Parameters**: {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}

### Random Forest:
- **Accuracy**: 86.12%
- **Precision (Class 0)**: 0.89, **Recall (Class 0)**: 0.94
- **Precision (Class 1)**: 0.73, **Recall (Class 1)**: 0.60

### Logistic Regression:
- **Accuracy**: 84.91%
- **Precision (Class 0)**: 0.88, **Recall (Class 0)**: 0.92
- **Precision (Class 1)**: 0.70, **Recall (Class 1)**: 0.58

## Insights

- **XGBoost** performed the best in terms of overall accuracy, precision, and recall.
- **Important Features**:
  - **marital-status_ Married-civ-spouse** and **education-num** were the most important predictors for income prediction.
  - **capital-gain** and **hours-per-week** also played significant roles in the predictions.
  
- **True Positive Prediction Focus**: I adjusted the model parameters to focus on improving recall for the higher income category (Class 1), aiming for more true positives.

## Conclusion

This project showcases how demographic features can be used to predict income levels. By comparing multiple machine learning models, I was able to identify **XGBoost** as the most effective model for this task. Further improvements could include fine-tuning hyperparameters and testing additional models.

## Future Work

- **Hyperparameter Tuning**: Experimenting with different hyperparameters for XGBoost and other models.
- **Cross-Validation**: Implementing cross-validation to ensure more robust model performance.
- **Feature Engineering**: Exploring additional features or feature transformations to improve model performance.

---

**Dataset Source**: UCI Machine Learning Repository - [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

