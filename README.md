# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using Logistic Regression and Random Forest.

The project covers exploratory data analysis, preprocessing, model training, cross-validation, independent test evaluation, and dataset shift analysis.

## Project Overview

Customer churn is an important business problem where companies aim to identify customers who are likely to stop using their services.

The objective of this project is to build classification models that predict customer churn and investigate the factors associated with churn behavior.

A key part of this project is the comparison between validation performance and performance on a separately supplied test dataset.

## Project Report

A detailed report covering the full analysis, model evaluation, and dataset shift investigation is available here:

[View the Full Project Report](reports/CustomerChurnMLProjectReport.pdf)


## Dataset

The dataset used in this project is the **Customer Churn Dataset** published by Muhammad Shahid Azeem on Kaggle.

It contains customer information such as:

- Age
- Gender
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Subscription Type
- Contract Length
- Total Spend
- Last Interaction
- Churn

The project uses the training and testing files supplied with the dataset.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── customer_churn_dataset-training-master.csv
│   └── customer_churn_dataset-testing-master.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_modeling.ipynb
│
├── reports/
│   └── CustomerChurnMLProjectReport.pdf
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

The exploratory analysis examined:

- Dataset dimensions and data types
- Missing values and duplicates
- Target class distribution
- Numerical feature distributions
- Categorical feature relationships with churn
- Churn rates by contract type
- Feature correlations

Some of the strongest relationships with churn in the training data were observed for:

- Support Calls
- Total Spend
- Payment Delay
- Age
- Contract Length

One particularly strong pattern was found in `Contract Length`: all customers with monthly contracts in the supplied training dataset were labeled as churned.

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of `CustomerID`
- Standardization of numerical features
- One-hot encoding of categorical features
- Handling of previously unseen categories using `handle_unknown="ignore"`
- Stratified train/validation splitting

The data was split into:

- 80% training data
- 20% validation data

## Models

Two classification models were evaluated:

### Logistic Regression

Validation performance:

| Metric | Score |
|---|---:|
| Accuracy | 89.34% |
| ROC-AUC | 0.9590 |

Removing `Contract Length` reduced validation accuracy to approximately **85.07%**, indicating that the feature contains significant predictive information.

### Random Forest

Validation performance:

| Metric | Score |
|---|---:|
| Accuracy | 99.93% |
| ROC-AUC | 1.0000 |

The model achieved approximately 100% training accuracy and 99.93% validation accuracy.

Because this performance was unusually high, additional tests were performed to investigate whether the result generalized beyond the validation split.

## Cross-Validation

Stratified 5-fold cross-validation was performed on the training dataset.

Average results:

| Metric | Score |
|---|---:|
| Accuracy | 99.92% |
| Precision | 1.0000 |
| Recall | 0.9987 |
| F1 Score | 0.9993 |
| ROC-AUC | 1.0000 |

The very small variation between folds showed that the Random Forest result was consistent across different subsets of the supplied training data.

## Independent Test Evaluation

The models were then evaluated on the separately supplied test dataset.

### Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 50.41% |
| Precision | 48.85% |
| Recall | 99.87% |
| F1 Score | 65.61% |
| ROC-AUC | 0.6123 |

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 57.12% |
| Precision | 52.51% |
| Recall | 99.06% |
| F1 Score | 68.64% |
| ROC-AUC | 0.6894 |

The large performance drop compared with validation indicated that the independent test population differs substantially from the training population.

## Dataset Shift Analysis

Further investigation revealed several differences between the supplied training and test datasets.

Examples include:

- Monthly contracts represent about 19.8% of the training data but about 34.4% of the test data.
- Average Support Calls increased from approximately 3.61 in training to 5.40 in testing.
- Average Payment Delay increased from approximately 12.97 to 17.13.
- Average Total Spend decreased from approximately 631.89 to 541.02.
- Relationships between important predictors and churn also changed between the two datasets.

This suggests both **covariate shift** and changes in feature-target relationships.

Therefore, the extremely high validation and cross-validation performance should not be interpreted as evidence that the model will generalize to a different customer population.

## Key Takeaways

This project demonstrates that strong validation metrics alone are not sufficient to establish model reliability.

The most important findings were:

- Random Forest achieved nearly perfect validation and cross-validation performance.
- Logistic Regression produced lower but still strong validation results.
- Both models experienced major performance degradation on the independent test dataset.
- Further analysis identified substantial differences between the training and test distributions.
- Independent testing and dataset-shift analysis were essential for revealing the model's generalization limitations.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks in the following order:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_modeling.ipynb
```

## Future Improvements

Potential next steps would focus on improving model reliability rather than deployment:

- Obtain a more representative and consistently sampled dataset
- Investigate the source of the distribution shift between training and test populations
- Evaluate additional models after resolving the dataset inconsistency
- Apply probability calibration and threshold analysis if reliable training data becomes available
- Monitor feature and prediction drift before considering production deployment

Due to the substantial performance degradation on the independent test set, the current models are not considered production-ready.

## Author

**Talal Alotaibi**

GitHub: [i7zk](https://github.com/i7zk)

LinkedIn: [Talal Alotaibi](https://www.linkedin.com/in/tbalotaibi/)