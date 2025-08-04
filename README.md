# Gaia Spectral Classification with Machine Learning
# Overview
This project tackles a data mining problem using the Gaia ESA Archive, aiming to predict the spectral class of celestial objects based on astrometry, photometry, and spectroscopy data. The workflow includes data preprocessing, model training, and evaluation using various classification algorithms.

# Project Description
Gaia is a European space mission collecting high-precision measurements for over 2 billion stars. The goal of this project is to predict the spectral class (SpType-ELS) using independent variables from Gaia's dataset. A total of 7 machine learning models were implemented, evaluated, and compared using metrics like accuracy, R-squared score, and cross-validation performance.

# Data Preprocessing
Key preprocessing steps:

- Column Removal: Dropped ID, Unnamed: 0, and Source as they were unique identifiers.

- Missing Value Handling: Dropped pscol due to excessive missing data.

- Imputed others with median values.

Feature Selection:

- Pearson correlation coefficient threshold of 0.4 used for feature selection.

Multicollinearity Reduction:

- Removed variables with correlation > 0.8.

- Final dataset reduced to 7 features.

Normalization:

- Used z-score normalization.

Target Variable:

- Binarized target variable for binary classification.

Train/Test Split:

- 80/20 split applied.

# Models Implemented
✅ Multilayer Perceptron (MLP)
- Two hidden layers: [100, 50]

- Optimizer: Adam

- Accuracy: 99.44% (holdout)

- R²: 0.978

- Cross-val accuracy: 96.9%

- Cross-val R²: 0.874

* A simpler configuration with one hidden layer performed similarly with faster training time.

🌳 Decision Tree
- Max depth: 4

- Accuracy: 98.9%

- R²: 0.958

- Cross-val accuracy: 95%

- Cross-val R²: 0.797

🌲 Random Forest
- Trees: 250

- Max depth: 4

- Accuracy: 99.3%

- R²: 0.974

- Cross-val accuracy: 95.3%

- Cross-val R²: 0.812

🤝 K-Nearest Neighbors (KNN)
- Best k: 8

- Accuracy: 99.2%

- R²: 0.969

- Cross-val accuracy: 97%

- Cross-val R²: 0.88

➕ Logistic Regression
- Accuracy: 99.1%

- R²: 0.964

- Cross-val accuracy: 98.7% (highest)

- Cross-val R²: 0.948

* Feature importance: Teff had the largest coefficient (18.4)

💠 Support Vector Machine (SVM)
- Kernel: RBF

- Accuracy: 99.3%

- R²: 0.971

- Cross-val accuracy: 97.6%

- Cross-val R²: 0.902

⚡ XGBoost
- Trees: 150

- Max depth: 5

- Learning rate: 0.1

- Accuracy: 99.6% (highest)

- R²: 0.983

- Cross-val accuracy: 96.5%

- Cross-val R²: 0.86

# Model Evaluation & Selection
Models were compared using:

- Holdout accuracy and R²

- 5-fold cross-validation accuracy and R²

- ROC Curve & AUC

Although XGBoost had the highest holdout scores, MLP and Logistic Regression performed better in cross-validation, which provides a more reliable measure. Hence, MLP and Logistic Regression were selected for final prediction.

# Prediction Results
The selected models were used to predict spectral class on the unknown dataset.

Final predictions were submitted to a Kaggle Competition, yielding:

- MLP: 0.99245

- Logistic Regression: 0.99132

# Discussion & Future Work
Lessons Learned:
- Simple preprocessing like median imputation and Pearson correlation worked well, but other techniques (e.g., KNN imputation, polynomial features) could improve performance.

- Manual hyperparameter tuning was effective, though using GridSearchCV would provide a more systematic approach.

- The similarity in model performance made it difficult to differentiate them using only ROC/AUC.

# Future Improvements:
- Use more robust imputation and feature engineering methods.

- Implement automated hyperparameter tuning.

- Evaluate additional metrics such as MAE and RMSE to better differentiate model performance.

# Author
Chen Hsin Lee

     



