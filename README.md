# 🧠 AI-Based Mental Health Risk Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Project Overview

Mental health disorders among students have become a significant concern due to increasing academic pressure, financial stress, lifestyle changes, and social challenges. Early identification of students at risk of depression can help institutions and healthcare professionals provide timely intervention and support.

This project presents a machine learning-based approach for predicting depression using demographic, academic, psychological, and lifestyle factors. Multiple supervised learning algorithms were trained and compared to identify the most effective predictive model. The project also incorporates hyperparameter tuning, threshold optimization, feature importance analysis, and cross-validation to improve model reliability and generalization.

Unlike many basic classification projects that rely solely on accuracy, this work focuses on achieving a balanced trade-off between precision and recall through threshold optimization, making the predictions more suitable for practical decision-making.

---

# 🎯 Objectives

- Predict whether a student is likely to experience depression.
- Compare multiple supervised machine learning algorithms.
- Identify the best-performing classification model.
- Improve prediction performance using hyperparameter tuning.
- Optimize the classification threshold based on the F1-score.
- Evaluate model robustness using Stratified K-Fold Cross Validation.
- Analyze the importance of different features influencing depression.

---

# 📊 Dataset

**Dataset Name:** Student Depression Dataset

**Source:** Kaggle

The dataset contains demographic, academic, lifestyle, and psychological attributes collected from students for depression prediction.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Records | 27,901 |
| Total Features | 18 |
| Target Variable | Depression |

### Features

- Gender
- Age
- City
- Profession
- Academic Pressure
- Work Pressure
- CGPA
- Study Satisfaction
- Job Satisfaction
- Sleep Duration
- Dietary Habits
- Degree
- Have you ever had suicidal thoughts?
- Work/Study Hours
- Financial Stress
- Family History of Mental Illness

**Target Variable**

- Depression (Yes / No)

---

# 🛠 Technologies Used

## Programming Language

- Python

## Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Joblib

---


            ▼
 Data Preprocessing
            │
            ▼
 Feature Encoding
            │
            ▼
 Feature Scaling
            │
            ▼
 Train-Test Split
            │
            ▼
 Multiple ML Models
            │
            ▼
 Model Comparison
            │
            ▼
 Hyperparameter Tuning
            │
            ▼
 Threshold Optimization
            │
            ▼
 Cross Validation
            │
            ▼
 Final Random Forest Model
            │
            ▼
 Performance Evaluation
```

---

# ⚙️ Data Preprocessing

Several preprocessing steps were performed to improve data quality before model training.

### Data Cleaning

- Removed unnecessary columns
- Checked missing values
- Handled null values
- Removed duplicate records
- Verified data consistency

### Feature Encoding

Categorical variables were converted into numerical representations using appropriate encoding techniques.

Examples include:

- Gender
- City
- Degree
- Dietary Habits
- Sleep Duration
- Family History of Mental Illness
- Suicidal Thoughts

### Feature Scaling

Numerical features were standardized before training models that are sensitive to feature magnitude, ensuring stable learning and improved convergence.

# 📈 Exploratory Data Analysis (EDA)

Before model development, an extensive exploratory data analysis was performed to understand the dataset, identify data quality issues, and discover relationships between variables.

The analysis included:

- Distribution of students with and without depression
- Gender-wise depression distribution
- Age distribution
- Academic pressure analysis
- Financial stress analysis
- Sleep duration analysis
- Dietary habits analysis
- Study satisfaction analysis
- Family history of mental illness
- Correlation analysis
- Feature importance visualization

The insights obtained from EDA helped identify the most influential variables affecting depression and guided the feature engineering process.

---

# 🤖 Machine Learning Models

Nine supervised machine learning algorithms were trained and evaluated on the same preprocessed dataset.

| Model | Purpose |
|--------|---------|
| Logistic Regression | Baseline Linear Classifier |
| Decision Tree | Tree-based Classification |
| Random Forest | Ensemble Learning |
| Gradient Boosting | Boosting Ensemble |
| Support Vector Machine (SVM) | Margin-based Classification |
| K-Nearest Neighbors (KNN) | Instance-based Learning |
| Naive Bayes | Probabilistic Classification |
| XGBoost | Gradient Boosted Decision Trees |
| LightGBM | Efficient Gradient Boosting |

Each model was evaluated using identical train-test splits and performance metrics to ensure a fair comparison.

---

# 📊 Model Evaluation Metrics

The following metrics were used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Precision-Recall Curve
- Cross Validation Score


---

# 🏆 Best Performing Model

After comparing all nine models, **Random Forest Classifier** achieved the best overall performance and was selected as the final model.

Reasons for selecting Random Forest include:

- Highest F1-score among all evaluated models
- Strong balance between Precision and Recall
- Robust performance on unseen data
- Lower tendency to overfit due to ensemble learning
- Reliable probability estimates for threshold optimization

---

# ⚡ Hyperparameter Tuning

To further improve model performance, **RandomizedSearchCV** was employed.

### Search Parameters

- Number of Estimators (`n_estimators`)
- Maximum Tree Depth (`max_depth`)
- Minimum Samples Split (`min_samples_split`)

### Best Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| n_estimators | 200 |
| max_depth | None |
| min_samples_split | 2 |

The tuned Random Forest achieved a Cross-Validation F1-score of:

**0.883**

---

# 🎯 Threshold Optimization

Instead of relying on the default classification threshold of **0.50**, threshold optimization was performed using the Precision-Recall Curve.

Three threshold strategies were evaluated:

### 1. Default Threshold (0.50)

- Accuracy: **84%**
- Balanced overall performance

### 2. F1-Optimal Threshold (Selected)

Threshold:

**0.435**

Performance:

| Metric | Value |
|--------|-------|
| Accuracy | **84%** |
| Precision (Class 1) | **0.84** |
| Recall (Class 1) | **0.93** |
| F1-Score (Class 1) | **0.88** |

This threshold provided the best balance between identifying depressed students and minimizing false positives.

---

### 3. Recall-Optimized Threshold

Threshold:

**0.220**

Performance:

- Recall increased to **98%**
- Precision decreased significantly
- False-positive rate increased considerably

Although this configuration identified nearly every depressed student, the large number of false positives made it unsuitable for practical deployment.

Therefore, the **F1-optimal threshold (0.435)** was selected as the final decision threshold.

---

# 🔄 Cross Validation

To evaluate the robustness and generalization capability of the final model, **5-Fold Stratified Cross Validation** was performed.

### Results

| Metric | Value |
|--------|-------|
| Mean Accuracy | **84.5%** |
| Mean F1 Score | **0.881** |
| Standard Deviation (F1) | **0.003** |

The low standard deviation across folds demonstrates that the model produces stable and consistent predictions on unseen data.

---

# 📋 Final Model Performance

## Final Model

**Random Forest Classifier**

## Final Threshold

**0.435**

### Classification Report

| Metric | Class 0 | Class 1 |
|---------|---------|---------|
| Precision | 0.85 | 0.84 |
| Recall | 0.69 | 0.93 |
| F1-Score | 0.76 | 0.88 |

### Overall Accuracy

**84%**

The final model successfully achieved a strong balance between precision and recall, making it well suited for early depression risk prediction.

---

# 📌 Feature Importance

Feature importance analysis was performed using the final tuned Random Forest model to identify the variables that contributed most to depression prediction.

| Rank | Feature | Importance |
|------|---------|-----------:|
| 1 | Have you ever had suicidal thoughts? | 0.215 |
| 2 | Academic Pressure | 0.163 |
| 3 | City (Combined) | 0.144 |
| 4 | Financial Stress | 0.095 |
| 5 | CGPA | 0.082 |
| 6 | Work/Study Hours | 0.074 |
| 7 | Age | 0.059 |
| 8 | Study Satisfaction | 0.045 |
| 9 | Dietary Habits | 0.034 |
| 10 | Sleep Duration | 0.034 |
| 11 | Degree | 0.022 |
| 12 | Gender | 0.017 |
| 13 | Family History of Mental Illness | 0.015 |

### Key Findings

The model identified psychological and academic factors as the strongest indicators of depression.

Major observations include:

- Previous suicidal thoughts were the strongest predictor.
- Higher academic pressure significantly increased depression risk.
- Financial stress also showed a strong relationship with depression.
- Lifestyle factors such as sleep duration and dietary habits influenced predictions.
- Study satisfaction and work/study hours were important academic indicators.

---

# 📊 Results Summary

| Category | Result |
|----------|---------|
| Dataset Size | 27,901 Records |
| Number of Features | 18 |
| Models Compared | 9 |
| Best Model | Random Forest |
| Hyperparameter Tuning | RandomizedSearchCV |
| Threshold Optimization | Precision-Recall Curve |
| Final Threshold | 0.435 |
| Test Accuracy | **84%** |
| Mean Cross Validation Accuracy | **84.5%** |
| Mean Cross Validation F1 Score | **0.881** |


---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/iqraafatima/AI-Based-Mental-Health-Risk-Prediction.git
```

Move into the project directory

```bash
cd AI-Based-Mental-Health-Risk-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```


Open the notebook and execute all cells sequentially.

---

# 📦 Required Libraries

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- joblib

---

# ▶️ Usage

1. Load the Student Depression Dataset.
2. Run the preprocessing pipeline.
3. Train and compare all machine learning models.
4. Perform RandomizedSearchCV for hyperparameter tuning.
5. Optimize the classification threshold.
6. Evaluate the final Random Forest model.
7. Analyze feature importance and model performance.

---

# 💡 Future Improvements

- Develop a real-time web application using Streamlit.
- Deploy the trained model on a cloud platform.
- Integrate Explainable AI techniques such as SHAP and LIME.
- Extend the dataset with additional demographic and behavioral features.
- Explore Deep Learning approaches for performance comparison.
- Build an early intervention recommendation system based on prediction outcomes.

---

# 📚 References

- Student Depression Dataset – Kaggle
- Scikit-learn Documentation
- XGBoost Documentation
- LightGBM Documentation

---

# 👩‍💻 Author

**Iqra Fatima Umang**

Master of Computer Applications (MCA)

Aligarh Muslim University

---

# ⭐ Acknowledgements

This project was developed as part of academic learning and research to explore the application of Machine Learning in mental health prediction. It demonstrates the complete machine learning pipeline, including data preprocessing, model comparison, hyperparameter tuning, threshold optimization, and performance evaluation for early depression risk assessment.

If you found this project helpful, consider giving the repository a ⭐ on GitHub.
