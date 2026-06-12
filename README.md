# 🚢 Titanic Survival Prediction — Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete machine learning project predicting Titanic passenger survival using **Logistic Regression** with a full sklearn `Pipeline`, `GridSearchCV` hyperparameter tuning, and extensive Exploratory Data Analysis (EDA).

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | ~81% |
| Test ROC-AUC | ~0.87 |
| CV ROC-AUC (5-fold) | ~0.86 |

---

## 📁 Project Structure

```
titanic-logistic-regression/
│
├── Titanic_Survival_Logistic_Regression.ipynb   # Main notebook
├── README.md                                     # Project documentation
└── requirements.txt                              # Dependencies
```

---

## 🔍 Notebook Sections

| Section | Description |
|---|---|
| 1. Imports & Data Loading | Libraries, dataset loading from Seaborn |
| 2. Dataset Overview | Shape, dtypes, missing values, target balance |
| 3. Exploratory Data Analysis | 9 sub-sections covering all key features |
| 4. Feature Engineering & Preprocessing | Column selection, sklearn Pipelines, train/test split |
| 5. Model Building | Logistic Regression pipeline + GridSearchCV |
| 6. Model Evaluation | Accuracy, ROC-AUC, confusion matrix, ROC curve, feature importance |
| 7. Conclusion | Summary table, key takeaways, next steps |

---

## 📈 Key EDA Insights

- **Gender is the strongest predictor:** Females survived at ~74% vs. ~19% for males — consistent with *"women and children first"* evacuation protocol.
- **Class matters significantly:** 1st class passengers survived at 63% while 3rd class had only 24% survival despite making up 55% of passengers.
- **Interaction effect:** A 1st-class female had a **96%** survival rate; a 3rd-class male had only **13%**.
- **Age signal:** Children had higher survival; seniors had the lowest odds.
- **Family size sweet spot:** Passengers traveling with 1–3 family members survived at higher rates than solo travelers or large groups.
- **Embarkation port:** Carries independent signal beyond class composition.

---

## 🛠️ Technical Highlights

- **End-to-end sklearn Pipeline** — ensures no data leakage between train and test sets.
- **Separate preprocessing sub-pipelines** for numeric (median imputation + StandardScaler), nominal categorical (mode imputation + OneHotEncoder), and ordinal categorical (`class` with OrdinalEncoder).
- **GridSearchCV** over regularisation strength `C ∈ {0.01, 0.1, 1, 10, 100}` with 5-fold KFold CV, optimising ROC-AUC.
- **Stratified train/test split** to preserve class balance.
- **Feature importance** visualised via logistic regression coefficients (log-odds).

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook Titanic_Survival_Logistic_Regression.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

See `requirements.txt` for pinned versions.

---

## 🏗️ Model Architecture

```
Input Features
     │
     ▼
ColumnTransformer
├── NumericPipeline    → [age, fare, sibsp, parch]     → MedianImputer → StandardScaler
├── NominalPipeline    → [sex, embarked]               → ModeImputer → OneHotEncoder
└── OrdinalPipeline    → [class]                       → OrdinalEncoder (First<Second<Third)
     │
     ▼
LogisticRegression (best C from GridSearchCV)
     │
     ▼
Binary Output: Survived (1) / Died (0)
```

---

## 📌 Future Work

- [ ] Try ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] Engineer `title` feature from passenger names
- [ ] Engineer `family_size = sibsp + parch` as a model feature
- [ ] Apply SHAP values for deeper explainability
- [ ] Address class imbalance with SMOTE or `class_weight='balanced'`

---

## 👤 Author

**Himanshu Jain**  
Master's in Data Science — University of Colorado Boulder  
[GitHub](https://github.com/himanshumjain15) · [Portfolio](https://himanshumjain15.github.io/Portfolio)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
