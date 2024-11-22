# Credit Risk Prediction Model

## Objective
To develop a machine learning model that predicts the likelihood of customer default using payment history and demographic data.

## Tools and Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`, `matplotlib.patches`
- **Data Preprocessing**: `LabelEncoder`, `OrdinalEncoder`, `StandardScaler`
- **Handling Class Imbalance**: `ADASYN` (from `imbalanced-learn`)
- **Model Development**:
  - **Classification Models**: `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`, `XGBClassifier` (from XGBoost)
  - **Feature Selection**: Random Forest feature importance
- **Model Evaluation**: `accuracy_score`, `f1_score`, `precision_score`, `recall_score`, `confusion_matrix`
- **Cross-Validation**: `cross_val_score`
- **Utility**: `collections.Counter`

## Dataset
The dataset sourced from Kaggle consists of two components:
- **Application Record**: 438,557 records with 18 features.
- **Credit Record**: 1,048,575 records with 3 features.

## Methodology
1. **Data Preprocessing**:
   - Handled missing values.
   - Encoded categorical features using `LabelEncoder` and `OrdinalEncoder`.
   - Addressed class imbalance using `ADASYN`.
   - Scaled features using `StandardScaler`.
2. **Feature Selection**:
   - Used Random Forest feature importance to identify key predictors for credit risk assessment.

## Models
- **Logistic Regression**
- **Decision Tree**
- **XGBoost**
- **Random Forest**

The Random Forest model demonstrated the best performance with:
- **Accuracy**: 98.20%
- **F1 Score**: 0.982

## Validation
- Performed 5-fold cross-validation with the Random Forest model.
- Achieved a **mean accuracy of 89.81%** with a low standard deviation of **0.0159**, confirming robustness and generalizability.

## Conclusion
The Random Forest model effectively classifies customers into low-risk and high-risk categories, enabling financial institutions to:
- Optimize credit decisions.
- Reduce defaults.
- Align with the business need for accurate credit risk prediction.
