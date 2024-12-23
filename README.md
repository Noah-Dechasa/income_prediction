# 📊 Adult Income Prediction

## 👨‍💻 Project Overview

The goal of this project was to predict whether an individual's income is greater than $50K per year based on various demographic and employment-related features. The dataset was sourced from the **UCI Machine Learning Repository** and includes features like age, education, occupation, and more.

### 🧩 Key Features of the Project:
- **Data Preprocessing**: Handling missing values, scaling numerical data, and one-hot encoding categorical variables.
- **Modeling**: Implemented multiple machine learning models including Logistic Regression, Random Forest, and XGBoost.
- **Evaluation**: Used metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
- **Hyperparameter Tuning**: Applied grid search for optimizing XGBoost's performance.
- **Feature Importance**: Analyzed the importance of different features for predicting income.

---

## 🔎 Goal

The primary objective of this project was to:
- **Predict** whether an individual earns more than $50K annually based on various attributes.
- **Understand** the most influential factors contributing to higher income predictions.
- **Evaluate** the performance of different machine learning models and compare their accuracy.

---

## 🛠 Tools & Libraries Used

- **Python**: The primary programming language used for this project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning models and preprocessing.
- **XGBoost**: For training the gradient boosting model.
- **Matplotlib / Seaborn**: For data visualization.
- **Jupyter Notebooks**: For experimentation and prototyping.

---

## 📊 Dataset

The dataset used is the **Adult Income Dataset**, available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult).

**Features include**:
- Age
- Education
- Marital Status
- Occupation
- Capital Gain/Loss
- Hours Worked per Week
- Native Country

**Target Variable**: `Income` (whether the person earns more than $50K annually).

---

## 🚀 Process

1. **Data Cleaning & Preprocessing**:
   - Handled missing values using `SimpleImputer`.
   - One-hot encoded categorical variables.
   - Scaled numerical features using `StandardScaler`.

2. **Modeling**:
   - Trained three different models: Logistic Regression, Random Forest, and XGBoost.
   - Applied cross-validation and grid search to fine-tune the XGBoost model.

3. **Evaluation**:
   - Measured accuracy, precision, recall, and F1-score to evaluate the model performance.
   - Identified the most important features affecting income predictions.

---

## 📈 Results

### Performance Comparison:

| Model            | Accuracy | Precision (<=50K) | Precision (>50K) | Recall (<=50K) | Recall (>50K) |
|------------------|----------|-------------------|------------------|----------------|---------------|
| Logistic Regression | 86.5%    | 0.90              | 0.78             | 0.85           | 0.73          |
| Random Forest     | 87.0%    | 0.89              | 0.79             | 0.89           | 0.70          |
| XGBoost (Tuned)  | 86.0%    | 0.92              | 0.71             | 0.90           | 0.75          |

**Best Performing Model**: Random Forest (87% Accuracy) but XGBoost (Tuned) better for Recall.

---

## 💡 Insights

- **Top Features Influencing Predictions**:
  - **Marital Status** (specifically, "Married-civ-spouse") was the most important feature for predicting income levels, contributing 52% to the model's decision-making process.
  - Other significant features include **Education Level**, **Capital Gain**, and **Occupation**.
  
- **Model Comparison**:
  - While **XGBoost** produced a higher precision for the higher-income class, **Random Forest** showed the best overall accuracy and recall balance.
  
- **Next Steps**:
  - Further exploration of **feature engineering** could improve model performance.
  - Experimenting with **neural networks** and **ensemble methods** might help improve accuracy.

---

## 🗂 Repository Structure

