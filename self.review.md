# Week 1 Self Review Checklist

- [x] Code runs successfully
- [x] Duplicate rows removed from student_scores.csv
- [x] Cleaned CSV file verified
- [x] Output evidence captured
- [x] Changes committed to Git branch
- [x] Changes pushed to GitHub
- [x] Code follows basic Python coding standards

## W1D2 Self Review

- [x] Loaded a real Indian dataset into a Pandas DataFrame.
- [x] Printed dataset shape, data types, and first 10 rows.
- [x] Performed filter operation.
- [x] Performed groupby operation.
- [x] Performed merge operation.
- [x] Created a pivot table.
- [x] Exported the dataset to CSV.
- [x] Exported the dataset to Parquet.
- [x] Compared CSV and Parquet file sizes.

### Reflection

Today I learned how to use Pandas for data analysis. I practiced loading a real dataset, exploring its structure, filtering data, grouping records, merging DataFrames, creating pivot tables, and exporting data to different file formats.

## W1D3 Self Review

- [x] Completed NumPy array operations.
- [x] Calculated array statistics (shape, mean, standard deviation, minimum, maximum).
- [x] Applied boolean masking to filter values above the average.
- [x] Performed matrix addition, multiplication, and transpose operations.
- [x] Implemented broadcasting for column-wise normalization.
- [x] Tested the program with 3 different input arrays.
- [x] Reviewed the code using CIA Full Stack Mentor Mode.
- [x] Applied CIA review suggestions (improved code readability and reproducibility).
- [x] Output evidence captured.
- [x] Changes committed and pushed to GitHub.

### Reflection

Today I learned how to perform fundamental NumPy operations that are widely used in machine learning. I practiced calculating array statistics, filtering data with boolean masking, performing matrix operations, and using broadcasting for normalization. I also improved my code by applying feedback from the CIA code review, making it more readable and reproducible.

## W1D4 Self Review

- [x] Loaded the Indian population dataset successfully.
- [x] Ran df.describe(), df.info(), and df.isnull().sum().
- [x] Documented 5 EDA observations.
- [x] Created numeric column distribution plots.
- [x] Created a correlation heatmap.
- [x] Created a top-10 category count plot using TRU.
- [x] Saved output evidence in output_evidence/.
- [x] Wrote a 200-word EDA narrative.
- [x] Reviewed the code using CIA Full Stack Mentor Mode.
- [x] Completed 2 CIA interactions.
- [x] Committed the W1D4 changes to the Git branch.
- [x] Pushed the W1D4 changes to GitHub.

### Reflection

Today I learned how to perform Exploratory Data Analysis using Pandas, Matplotlib, and Seaborn. I practiced examining descriptive statistics, dataset information, and missing values. I also created numeric distribution plots, a correlation heatmap, and a top-category count plot to understand patterns in the Indian population dataset. The CIA Full Stack Mentor reviews helped me identify areas for improving code quality and reproducibility. This task improved my understanding of how data is explored and prepared before applying machine learning models.

## W1D5 Self Review

- [x] Loaded the Indian population dataset successfully.
- [x] Created a top-10 district population bar chart.
- [x] Created a population distribution histogram.
- [x] Created a male vs female population scatter plot.
- [x] Created a population correlation heatmap.
- [x] Saved all visualization output evidence in output_evidence/w1d5_plots/.
- [x] Used Matplotlib and Seaborn for data visualisation.
- [x] Reviewed the code using CIA Full Stack Mentor Mode.
- [x] Applied CIA review suggestions for code quality and validation.
- [x] Successfully tested the visualization script.
- [x] Committed the W1D5 changes to the Git branch.
- [x] Pushed the W1D5 changes to GitHub.
- [x] Added W1D5 changes to the Pull Request.

### Reflection

Today I learned how to transform raw population data into meaningful visual insights using Matplotlib and Seaborn. I created bar, histogram, scatter, and correlation heatmap visualisations to understand population rankings, distributions, relationships, and correlations. I also improved the code based on CIA review feedback by adding validation, a reusable plotting helper, and a main function. This task strengthened my understanding of data visualisation as an important step in EDA before machine learning.

## W2D1 Self Review

- [x] Applied LabelEncoder to a categorical column.
- [x] Applied OneHotEncoder to a categorical column.
- [x] Applied OrdinalEncoder to a categorical column.
- [x] Documented encoding trade-offs.
- [x] Applied StandardScaler.
- [x] Applied MinMaxScaler.
- [x] Applied RobustScaler.
- [x] Created scaling comparison visualization.
- [x] Used SelectKBest to identify the top 5 features.
- [x] Documented why the top 5 features matter.
- [x] Saved encoding results as output evidence.
- [x] Saved scaling visualization as output evidence.
- [x] Saved top 5 feature scores as output evidence.
- [x] Reviewed the code using CIA Full Stack Mentor Mode.
- [x] Applied CIA review improvements.
- [x] Completed final CIA review with READY FOR SUBMISSION verdict.
- [x] Committed W2D1 implementation.
- [x] Working on the Week 2 branch.

### Top 5 Feature Importance

- **TOT_M:** Represents total male population and contributes directly to total population.
- **TOT_F:** Represents total female population and contributes directly to total population.
- **NON_WORK_P:** Represents the non-working population and describes population composition.
- **TOT_WORK_P:** Represents the working population and captures an important population characteristic.
- **P_LIT:** Represents the literate population and provides an indicator of literacy.

### Reflection

Today I learned how feature engineering prepares data for machine learning. I practiced categorical encoding using LabelEncoder, OneHotEncoder, and OrdinalEncoder and compared their trade-offs. I also applied StandardScaler, MinMaxScaler, and RobustScaler to understand different scaling methods. Finally, I used SelectKBest to identify the five most relevant features. CIA feedback helped me improve code organization, reduce repetition, and document methodological limitations.

## W2D2 Self Review

- [x] Applied LabelEncoder to a categorical column.
- [x] Applied OneHotEncoder to a categorical column.
- [x] Applied OrdinalEncoder to a categorical column.
- [x] Documented encoding trade-offs.
- [x] Applied StandardScaler to numeric predictor features.
- [x] Applied MinMaxScaler to numeric predictor features.
- [x] Applied RobustScaler to numeric predictor features.
- [x] Created before/after scaling distribution plots.
- [x] Used SelectKBest with `f_regression` to identify the top 5 features for numeric target `TOT_P`.
- [x] Documented why the selected top 5 features matter.
- [x] Generated and verified output evidence.
- [x] Completed CIA Full Stack Mentor review interaction 1.
- [x] Completed CIA Full Stack Mentor review interaction 2.
- [x] Applied CIA review feedback to the scaling implementation.
- [x] Tested the updated code successfully.
- [x] Committed the W2D2 feature engineering and scaling implementation.
- [x] Created the second required commit.
- [x] Pushed changes to GitHub.
- [x] Raised/updated the Pull Request.

### W2D2 Top 5 Features

- **TOT_M:** Total male population is a major component of total population.
- **TOT_F:** Total female population is a major component of total population.
- **NON_WORK_P:** Represents the non-working population and provides information about population composition.
- **TOT_WORK_P:** Represents the total working population and provides information about the working population.
- **P_LIT:** Represents the literate population and provides information about literacy within the population.

### Reflection

Today I learned how feature scaling and feature selection can prepare numerical data for machine learning. I applied StandardScaler, MinMaxScaler, and RobustScaler to numeric predictor features and compared their distributions before and after scaling. I also used SelectKBest with f_regression to identify the five features with the strongest linear relationships with the numeric target TOT_P. CIA review helped me identify that the target variable should not be used for the scaling demonstration and guided me to apply the scalers to predictor features instead. I tested the corrected implementation successfully and verified the generated output evidence.

## W2D3 Self Review

- [x] Loaded the Indian population dataset successfully.
- [x] Checked the class distribution of the target variable.
- [x] Split the data into training and testing sets before applying SMOTE.
- [x] Applied SMOTE only to the training data to prevent data leakage.
- [x] Used `k_neighbors=1` because the minority class had only 2 training samples.
- [x] Verified class distribution before SMOTE.
- [x] Verified class distribution after SMOTE.
- [x] Created before-SMOTE class distribution evidence.
- [x] Created SMOTE class distribution evidence.
- [x] Saved the SMOTE-resampled dataset as output evidence.
- [x] Wrote clean and commented code.
- [x] Successfully tested the W2D3 script.
- [x] Reviewed the implementation using CIA Full Stack Mentor Mode.
- [x] Applied CIA review suggestions.
- [x] Completed the required CIA review interactions.
- [x] Committed the W2D3 implementation.
- [x] Pushed the W2D3 implementation to the Week 2 branch.

### Reflection

Today I learned how SMOTE can be used to handle imbalanced classification data. I checked the original and training class distributions, split the data before oversampling, and applied SMOTE only to the training set to prevent data leakage. Because the minority class contained only two training samples, I used `k_neighbors=1`. I also generated evidence showing the class distribution before and after SMOTE. The implementation was reviewed using CIA Full Stack Mentor Mode, and the suggested improvements were applied before committing and pushing the code.

## W2D4 Self Review

- [x] Loaded the Indian population dataset successfully.
- [x] Prepared numeric predictor features and separated the target variable.
- [x] Performed an 80/20 train/test split.
- [x] Used quantile-based target bins for stratified train/test splitting.
- [x] Used random_state=42 for reproducibility.
- [x] Applied StandardScaler inside a Pipeline to prevent data leakage.
- [x] Performed 5-fold cross-validation on the training data only.
- [x] Evaluated the model using RÂ², MAE, and RMSE.
- [x] Evaluated the final model on the unseen test set.
- [x] Generated train/test target distribution evidence.
- [x] Saved cross-validation results as CSV evidence.
- [x] Saved train/test and cross-validation summary as CSV evidence.
- [x] Tested the W2D4 script successfully.
- [x] Reviewed the implementation using CIA Full Stack Mentor Mode.
- [x] Applied CIA review considerations regarding data leakage and reproducibility.
- [x] Committed the W2D4 implementation.
- [x] Pushed the W2D4 implementation to the Week 2 branch.

## W2D5 Self Review

- [x] Loaded the Titanic dataset successfully.
- [x] Performed basic EDA and checked missing values.
- [x] Generated and saved EDA distribution evidence.
- [x] Explicitly handled unused columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- [x] Treated `Pclass` as a categorical/ordinal feature.
- [x] Separated target variable `Survived` from predictor features.
- [x] Performed an 80/20 train/test split.
- [x] Used `RANDOM_STATE = 42` for reproducibility.
- [x] Applied missing-value imputation using a preprocessing pipeline.
- [x] Applied `StandardScaler` to numeric features.
- [x] Applied `OneHotEncoder` to categorical features with `handle_unknown="ignore"`.
- [x] Fitted the preprocessing pipeline only on the training data to prevent data leakage.
- [x] Transformed the test data using the training-fitted pipeline.
- [x] Generated the ML-ready dataset successfully.
- [x] Verified that the ML-ready dataset contains no missing values.
- [x] Verified that the `Survived` target column is present.
- [x] Saved the ML-ready CSV as output evidence.
- [x] Saved the fitted preprocessing pipeline as a `.joblib` file.
- [x] Added a module-level docstring and comments for major processing steps.
- [x] Reviewed the implementation using CIA Full Stack Mentor Mode.
- [x] Completed the required 2 CIA review interactions.
- [x] Applied CIA review suggestions and corrected the implementation.
- [x] Successfully tested the final W2D5 script.
- [x] Committed the W2D5 implementation to the Week 2 branch.
- [x] Pushed the W2D5 implementation to GitHub.

### Reflection

Today I learned how to build an end-to-end preprocessing pipeline for machine learning using the Titanic dataset. I practiced EDA, missing-value imputation, categorical encoding, feature scaling, and train/test splitting. I also learned how fitting preprocessing only on training data prevents data leakage. The CIA reviews helped me improve the treatment of `Pclass`, explicitly handle unused columns, ensure reproducibility, and persist the preprocessing pipeline. I verified the final ML-ready output and saved the required evidence and preprocessing artifact.

# Week 3 Self Review

## W3D1 Self Review

- [x] Trained LinearRegression on the California Housing dataset.
- [x] Printed model coefficients and intercept.
- [x] Evaluated the LinearRegression model using MSE.
- [x] Evaluated the LinearRegression model using RMSE.
- [x] Evaluated the LinearRegression model using MAE.
- [x] Evaluated the LinearRegression model using RÂ².
- [x] Created the predicted vs actual plot.
- [x] Created the residual plot.
- [x] Added Ridge Regression.
- [x] Added Lasso Regression.
- [x] Compared LinearRegression, Ridge, and Lasso models.
- [x] Saved model comparison results as CSV evidence.
- [x] Saved predicted vs actual plot as output evidence.
- [x] Saved residual plot as output evidence.
- [x] Reviewed the implementation using CIA Full Stack Mentor Mode.
- [x] Applied CIA review suggestions.
- [x] Successfully tested the W3D1 script.
- [x] Committed the W3D1 implementation.
- [x] Created the required second descriptive commit.
- [x] Pushed the W3D1 changes to the Week 3 branch.
- [x] Raised/updated the W3D1 Pull Request.

### Reflection

Today I learned how to build and evaluate regression models using Scikit-Learn. I trained LinearRegression on a real dataset and evaluated it using MSE, RMSE, MAE, and RÂ². I also learned how predicted-vs-actual and residual plots help evaluate regression performance. Finally, I compared LinearRegression with Ridge and Lasso regression and documented the results as output evidence. CIA review helped me verify the implementation and improve the overall quality of the W3D1 work.
##W3D2 self review
reflection
Today I learned how to build and evaluate a multiclass classification model using Logistic Regression and the Iris dataset. I practiced preparing the data, performing a stratified train/test split, scaling features without causing data leakage, and training a Logistic Regression classifier. I evaluated the model using accuracy, precision, recall, and F1-score and generated a classification report and confusion matrix to understand the model's performance. I also created a decision boundary visualization to understand how the classifier separates different Iris classes. The CIA Full Stack Mentor reviews helped me validate the implementation, improve code quality, and ensure the solution was ready for submission.

##W3D3 self review
Created the first W3D3 descriptive Git commit.

Created the second required W3D3 descriptive Git commit.

Pushed the W3D3 changes to the Week 3 branch.

Raised/updated the W3D3 Pull Request.

Reflection

Today I learned how Decision Tree classifiers can be used for multiclass classification using the Iris dataset. I trained Decision Trees using both Gini impurity and entropy/information gain and compared their training and testing performance. Both models achieved a testing accuracy of 0.9333 on the selected test set. I also examined tree depth and the number of leaves to understand model complexity and generated a visualization of the trained Decision Tree. Validation tests were added to verify the implementation, and the final script passed all validation tests. The implementation was reviewed and improved for code quality, reproducibility, and validation.

W3D4 Results
SVM
Best parameters: C=0.1, kernel=linear, gamma=scale
Cross-validation accuracy: 0.975
Test accuracy: 0.9333
Precision: 0.9333
Recall: 0.9333
F1-score: 0.9333
KNN
Best parameters: n_neighbors=5, weights=uniform, metric=euclidean
Cross-validation accuracy: 0.9667
Test accuracy: 0.9333
Precision: 0.9444
Recall: 0.9333
F1-score: 0.9327
Model Comparison

Both SVM and KNN achieved a test accuracy of 93.33% on the selected test set.

KNN achieved slightly higher precision, while SVM achieved a slightly higher F1-score.

When to Use What

SVM is useful when a strong classification boundary is required, particularly for datasets with high-dimensional feature spaces or cases where kernel methods can model complex decision boundaries.

KNN is useful when similar observations tend to have similar labels and when a simple, intuitive classification method is appropriate. Because KNN relies on distance calculations, feature scaling is important and prediction can become more computationally expensive as the dataset grows.

Reflection

Today I learned how Support Vector Machine and K-Nearest Neighbors classifiers can be used for multiclass classification using the Iris dataset. I practiced applying feature scaling through Scikit-Learn Pipelines and learned how hyperparameter tuning with GridSearchCV can be used to select suitable model configurations. I evaluated both models using accuracy, precision, recall, and F1-score and generated confusion matrices and comparison evidence to understand their performance. I also learned that SVM and KNN have different strengths and should be selected based on dataset characteristics and requirements. The implementation was tested successfully and reviewed for code quality, reproducibility, and prevention of data leakage.

W3D5 Results

SVM â€” GridSearchCV

Best CV accuracy: 0.975
Test accuracy: 0.9333

SVM â€” RandomizedSearchCV

Best CV accuracy: 0.975
Test accuracy: 0.9333

KNN â€” GridSearchCV

Best CV accuracy: 0.9667
Test accuracy: 0.9333

KNN â€” RandomizedSearchCV

Best CV accuracy: 0.9667
Test accuracy: 0.9667
Reflection

Today I learned how hyperparameter tuning can improve the process of selecting suitable machine learning model configurations. I implemented both GridSearchCV and RandomizedSearchCV for SVM and KNN using the Iris dataset. I used 5-fold cross-validation and placed feature scaling inside Scikit-Learn Pipelines to prevent data leakage. I compared the best cross-validation scores and evaluated the tuned models on an unseen test set using accuracy, precision, recall, and F1-score. The results showed that KNN with RandomizedSearchCV achieved the highest test accuracy of 96.67% on the selected test set. I also generated CSV and visualization evidence and verified the implementation for reproducibility and code quality.

## W4D1 - Model Evaluation

### Task Completed
Implemented model evaluation using the Breast Cancer Wisconsin dataset and Logistic Regression.

### Evaluation Metrics
- Precision: 0.9595
- Recall: 0.9861
- ROC-AUC: 0.9954

### Checklist
- [x] Loaded Breast Cancer Wisconsin dataset
- [x] Performed train/test split
- [x] Trained Logistic Regression model
- [x] Calculated Precision
- [x] Calculated Recall
- [x] Calculated ROC-AUC
- [x] Generated ROC curve
- [x] Saved evaluation evidence
- [x] Verified script runs successfully

### Evidence
- `output_evidence/w4d1/evaluation_metrics.txt`
- `output_evidence/w4d1/roc_curve.png`

### Verification
W4D1 script executed successfully without errors and the working tree was verified to be clean after the implementation commit.

## W4D2 - Bias-Variance Tradeoff & Regularisation

### Task Completed

Implemented Linear Regression, Ridge Regression, and Lasso Regression using the California Housing dataset. Applied systematic hyperparameter tuning using GridSearchCV and RandomizedSearchCV with 5-fold cross-validation. Demonstrated the effect of different Ridge regularisation strengths on training and validation error to study the bias-variance tradeoff.

### Model Results

| Model                    |      MSE |     RMSE |      MAE |       RÂ² |
| ------------------------ | -------: | -------: | -------: | -------: |
| Linear Regression        | 0.555892 | 0.745581 | 0.533200 | 0.575788 |
| Ridge GridSearchCV       | 0.555891 | 0.745581 | 0.533200 | 0.575788 |
| Lasso GridSearchCV       | 0.555745 | 0.745483 | 0.533192 | 0.575900 |
| Ridge RandomizedSearchCV | 0.555892 | 0.745581 | 0.533200 | 0.575788 |
| Lasso RandomizedSearchCV | 0.555309 | 0.745191 | 0.533172 | 0.576232 |

### Best Hyperparameters

* Ridge GridSearchCV: `alpha = 0.01`
* Lasso GridSearchCV: `alpha = 0.0001`
* Ridge RandomizedSearchCV: `alpha = 0.001`
* Lasso RandomizedSearchCV: `alpha â‰ˆ 0.0004037`



### Evidence

* `output_evidence/w4d2/model_comparison.csv`
* `output_evidence/w4d2/model_comparison_r2.png`
* `output_evidence/w4d2/ridge_grid_search_results.csv`
* `output_evidence/w4d2/lasso_grid_search_results.csv`
* `output_evidence/w4d2/ridge_randomized_search_results.csv`
* `output_evidence/w4d2/lasso_randomized_search_results.csv`
* `output_evidence/w4d2/bias_variance_results.csv`
* `output_evidence/w4d2/bias_variance_tradeoff.png`

### Verification

W4D2 script executed successfully without errors. The implementation was tested using the California Housing dataset, and the generated model comparison, hyperparameter search, and bias-variance evidence files were verified.

### Reflection

Today I learned how regularisation can help control model complexity and how Ridge and Lasso use different approaches to penalise model coefficients. I learned that GridSearchCV evaluates predefined hyperparameter combinations systematically, while RandomizedSearchCV samples configurations from a larger search space. I also practiced using Scikit-Learn Pipelines with StandardScaler to ensure preprocessing is performed correctly during cross-validation and to help prevent data leakage. The results showed that Lasso with RandomizedSearchCV achieved the best test-set performance among the evaluated models, although the improvement over the baseline was small. I also observed how increasing Ridge regularisation strength can increase both training and validation error, illustrating the effect of stronger regularisation on model bias and complexity.

## W4D3 - Model Serialisation

### Task Completed

Implemented model serialisation using Joblib and Pickle with a Linear Regression model trained on the California Housing dataset.

### Checklist

- [x] Loaded the California Housing dataset.
- [x] Performed train/test split.
- [x] Trained Linear Regression model.
- [x] Evaluated the original model using MSE, RMSE, MAE, and RÂ².
- [x] Serialized the trained model using Joblib.
- [x] Serialized the trained model using Pickle.
- [x] Loaded the Joblib model successfully.
- [x] Loaded the Pickle model successfully.
- [x] Verified restored-model predictions against the original model.
- [x] Confirmed Joblib predictions match the original predictions.
- [x] Confirmed Pickle predictions match the original predictions.
- [x] Compared original and restored model metrics.
- [x] Saved serialization evidence.
- [x] Successfully tested the W4D3 script.
- [x] Wrote clean and commented code.

### Model Results

| Model | MSE | RMSE | MAE | RÂ² |
|---|---:|---:|---:|---:|
| Original Linear Regression | 0.555892 | 0.745581 | 0.533200 | 0.575788 |
| Joblib Restored Model | 0.555892 | 0.745581 | 0.533200 | 0.575788 |
| Pickle Restored Model | 0.555892 | 0.745581 | 0.533200 | 0.575788 |

### Serialization Verification

- Joblib predictions match original: `True`
- Pickle predictions match original: `True`

### Evidence

- `output_evidence/w4d3/linear_regression_model.joblib`
- `output_evidence/w4d3/linear_regression_model.pkl`
- `output_evidence/w4d3/serialization_model_comparison.csv`
- `output_evidence/w4d3/serialization_prediction_evidence.csv`
- `output_evidence/w4d3/serialization_verification.csv`

### Reflection

Today I learned how trained machine learning models can be saved and restored using Joblib and Pickle. I trained a Linear Regression model on the California Housing dataset, evaluated its performance, serialized it using both methods, and loaded the saved models back into memory. I verified that the restored models produced the same predictions and evaluation metrics as the original model. This demonstrated how model serialization can preserve a trained model for later use without retraining it. I also practiced generating reproducible evidence and organizing model artifacts for future deployment workflows.

### Verification

The W4D3 script executed successfully without errors. Joblib and Pickle restoration were verified by comparing their predictions with the original model, and all generated evidence files were successfully created.


## W4D4 - FastAPI Model Serving Endpoint

### Task Completed

Implemented a FastAPI model-serving application that loads the serialized Linear Regression model created during W4D3 and provides health-check and prediction endpoints.

### Checklist

* [x] Loaded the serialized Linear Regression model from W4D3 using Joblib.
* [x] Created a FastAPI application.
* [x] Defined the California Housing model input schema using Pydantic.
* [x] Implemented the `/health` endpoint.
* [x] Implemented the `/predict` POST endpoint.
* [x] Added input validation for all 8 model features.
* [x] Used NumPy to prepare prediction input data.
* [x] Returned the predicted `MedHouseVal` value from the API.
* [x] Tested the FastAPI application successfully.
* [x] Verified that the model loads correctly.
* [x] Tested the `/health` endpoint successfully.
* [x] Tested the `/predict` endpoint successfully.
* [x] Verified the `/docs` Swagger documentation endpoint.
* [x] Verified `/docs` returned HTTP status code 200.
* [x] Used FastAPI and Uvicorn for model serving.
* [x] Removed generated `__pycache__` files before committing.
* [x] Wrote clean and commented code.
* [x] Created the first descriptive W4D4 implementation commit.
* [x] Created the second descriptive W4D4 self-review commit.
* [x] Pushed the W4D4 changes to GitHub.
* [ ] Raised/updated the W4D4 Pull Request.

### API Testing Evidence

#### Health Endpoint

The `/health` endpoint was tested successfully and returned:

* Status: `healthy`
* Model loaded: `True`
* Model: `Linear Regression`

#### Prediction Endpoint

The `/predict` endpoint was tested successfully using all eight California Housing features and returned a prediction for the target variable `MedHouseVal`.

#### Documentation Endpoint

The `/docs` endpoint was verified successfully and returned HTTP status code `200`.

### Reflection

Today I learned how to serve a trained machine learning model through a REST API using FastAPI and Uvicorn. I loaded the Linear Regression model serialized during W4D3 using Joblib and created endpoints for health checking and prediction. I used Pydantic to validate incoming request data and ensured that all eight California Housing features are provided before making a prediction. I also tested the API using PowerShell requests and verified the automatically generated Swagger documentation. This task helped me understand how a trained machine learning model can be exposed as an API for integration with other applications.

## W4D5 - 1M Capstone: Sentiment Classifier

### Task Completed

Implemented a binary sentiment classification pipeline using TF-IDF feature extraction, Logistic Regression, and Random Forest. Compared both models using accuracy, precision, recall, and ROC-AUC, and generated evaluation evidence.

### Checklist

- [x] Created a binary sentiment classification dataset.
- [x] Used 1,000 sentiment samples with balanced positive and negative classes.
- [x] Split the dataset into training and testing sets.
- [x] Applied TF-IDF vectorization to the text data.
- [x] Trained a Logistic Regression classifier.
- [x] Printed the Logistic Regression classification report.
- [x] Generated the Logistic Regression confusion matrix.
- [x] Generated the Logistic Regression ROC-AUC curve.
- [x] Calculated Logistic Regression accuracy.
- [x] Calculated Logistic Regression precision.
- [x] Calculated Logistic Regression recall.
- [x] Calculated Logistic Regression ROC-AUC.
- [x] Trained a Random Forest classifier.
- [x] Printed the Random Forest classification report.
- [x] Generated the Random Forest confusion matrix.
- [x] Calculated Random Forest accuracy.
- [x] Calculated Random Forest precision.
- [x] Calculated Random Forest recall.
- [x] Calculated Random Forest ROC-AUC.
- [x] Compared Logistic Regression and Random Forest performance.
- [x] Saved model comparison results.
- [x] Saved prediction evidence.
- [x] Created all required evaluation evidence files.
- [x] Tested the W4D5 script successfully without errors.
- [x] Wrote clean and commented code.
- [x] Created the first descriptive W4D5 implementation commit.
- [ ] Created the second descriptive W4D5 self-review commit.
- [ ] Pushed the W4D5 changes to GitHub.
- [ ] Raised/updated the W4D5 Pull Request.

### Model Results

Both models achieved the following results on the test set:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The dataset used for this task is a controlled sentiment dataset, so the perfect test scores should not be interpreted as proof of production-level performance.

### Evidence Generated

- `output_evidence/w4d5/classification_metrics.csv`
- `output_evidence/w4d5/logistic_confusion_matrix.png`
- `output_evidence/w4d5/logistic_roc_auc_curve.png`
- `output_evidence/w4d5/random_forest_confusion_matrix.png`
- `output_evidence/w4d5/model_comparison.csv`
- `output_evidence/w4d5/roc_auc_comparison.png`
- `output_evidence/w4d5/prediction_evidence.csv`
- `output_evidence/w4d5/sentiment_dataset.csv`

### Reflection

Today I learned how to build a complete binary sentiment classification pipeline using TF-IDF and machine learning classifiers. I trained and evaluated Logistic Regression and Random Forest models and compared their accuracy, precision, recall, and ROC-AUC scores. I also generated confusion matrices and ROC-AUC visualizations to understand model performance. This task helped me understand how text data can be converted into numerical features using TF-IDF and then used for supervised classification. I also learned that very high evaluation scores on a controlled dataset should be interpreted carefully and may not represent performance on real-world data.
## W5D1 - Running LLMs Locally with Ollama

### Task Completed

Set up Ollama for local LLM inference, pulled the Llama 3.2 3B and Qwen 2.5 3B models, created a Python script using the Ollama API with a custom system prompt, tested five AI/ML questions, and compared both models using the same three questions.

### Checklist

* [x] Installed and verified Ollama.
* [x] Pulled `llama3.2:3b` successfully.
* [x] Ran the first local LLM inference successfully.
* [x] Created `w5d1_ollama_inference.py`.
* [x] Used a custom system prompt for the local LLM.
* [x] Tested five AI/ML prompts using the Python script.
* [x] Confirmed all five prompts completed successfully.
* [x] Pulled `qwen2.5:3b` successfully.
* [x] Compared Llama 3.2 3B and Qwen 2.5 3B using the same three questions.
* [x] Documented differences between the two models.
* [x] Created W5D1 output evidence.
* [x] Tested the Python script successfully without errors.
* [x] Used clean and commented code.
* [ ] Completed CIA Full Stack Mentor code review.
* [ ] Created the first descriptive W5D1 commit.
* [ ] Created the second descriptive W5D1 self-review commit.
* [ ] Pushed W5D1 changes to GitHub.
* [ ] Raised/updated the W5D1 Pull Request.

### Models Compared

* `llama3.2:3b`
* `qwen2.5:3b`

### Comparison Summary

Based on the three comparison questions, Llama 3.2 3B produced responses that were generally more conversational and beginner-friendly, with simple examples and explanations.

Qwen 2.5 3B produced more structured and detailed responses and included more technical terminology and techniques.

For the beginner-focused system prompt used in this task, Llama 3.2 3B was easier to read, while Qwen 2.5 3B was useful when more technical detail was preferred.

### Evidence Generated

* `w5d1_ollama_inference.py`
* `output_evidence/w5d1/model_comparison.md`
* Terminal output copied into the W5D1 evidence document.

### Reflection

Today I learned how to run large language models locally using Ollama and interact with them through the Ollama API. I created a Python script with a custom system prompt and tested it using five AI/ML questions. I also compared Llama 3.2 3B and Qwen 2.5 3B using the same questions. The comparison showed that different local LLMs can produce different response styles and levels of detail even when given the same prompts. This task helped me understand the basics of local LLM inference and API-based interaction with locally hosted models. 

## W5D2 - Local LLM Model Comparison and Prompt Engineering

### Task Completed

Implemented local LLM prompt engineering and model comparison using Ollama. Tested a custom system prompt with five AI/ML questions and compared `llama3.2:3b` and `qwen2.5:3b` using the same three questions. Generated and verified output evidence documenting the qualitative differences between the models.

### Checklist

* [x] Verified Ollama local LLM setup.
* [x] Used `llama3.2:3b` for local inference.
* [x] Used `qwen2.5:3b` for model comparison.
* [x] Created `w5d2_prompt_engineering.py`.
* [x] Created `w5d2_model_comparison.py`.
* [x] Used a custom system prompt.
* [x] Tested five AI/ML prompts.
* [x] Compared both models using the same three questions.
* [x] Documented qualitative differences between the models.
* [x] Generated W5D2 output evidence.
* [x] Verified the W5D2 evidence files.
* [x] Tested the W5D2 implementation successfully.
* [x] Created the W5D2 implementation commit.
* [x] Pushed the W5D2 changes to GitHub.
* [ ] Completed CIA Full Stack Mentor code review.
* [ ] Applied CIA review suggestions, if applicable.
* [ ] Raised/updated the W5D2 Pull Request.

### Evidence Generated

* `w5d2_prompt_engineering.py`
* `w5d2_model_comparison.py`
* `output_evidence/w5d2/inference_results.txt`
* `output_evidence/w5d2/model_comparison.md`
* `output_evidence/w5d2/model_comparison_results.txt`

### Reflection

Today I learned how to interact with locally hosted large language models using Ollama and a custom system prompt. I practiced prompt engineering by testing five AI/ML questions and compared Llama 3.2 3B with Qwen 2.5 3B using the same three questions. The comparison showed differences in response style, structure, level of detail, and use of examples. I also learned how to document qualitative model comparisons and organize reproducible output evidence.


## W5D3 - ChromaDB Vector Store Setup and PDF RAG

### Task Completed

Implemented a ChromaDB vector store with 20 AI/ML documents, cosine similarity search, metadata filtering, and a PDF-based RAG workflow using ChromaDB and Ollama.

### Checklist

- [x] Installed ChromaDB and verified version 1.5.9.
- [x] Created a persistent ChromaDB collection.
- [x] Configured cosine similarity.
- [x] Added 20 AI/ML documents with embeddings.
- [x] Performed similarity search.
- [x] Performed metadata filtering using topic = classification.
- [x] Manually verified similarity search results.
- [x] Manually verified metadata filtering results.
- [x] Created an AI/ML reference PDF for the RAG workflow.
- [x] Extracted text from the PDF using pypdf.
- [x] Split the PDF text into 5 chunks.
- [x] Stored the PDF chunks in ChromaDB.
- [x] Retrieved the top-3 PDF chunks using cosine similarity.
- [x] Passed the retrieved chunks to Ollama.
- [x] Generated an answer using the retrieved PDF context.
- [x] Verified that the final answer was grounded in the retrieved context.
- [x] Generated W5D3 output evidence.
- [x] Created manual verification evidence.
- [x] Tested the W5D3 implementation successfully.
- [x] Created the first descriptive W5D3 implementation commit.
- [ ] Completed CIA Full Stack Mentor code review.
- [ ] Applied CIA review suggestions, if applicable.
- [ ] Created the second descriptive W5D3 self-review commit.
- [ ] Pushed the W5D3 changes to GitHub.
- [ ] Raised/updated the Pull Request.

### Evidence Generated

- output_evidence/w5d3/chromadb_setup_results.txt
- output_evidence/w5d3/pdf_rag_results.txt
- output_evidence/w5d3/manual_verification.md
- data/w5d3/ai_ml_reference.pdf

### Reflection

Today I learned how vector databases can be used to store and retrieve information using embeddings and similarity search. I created a ChromaDB collection containing 20 AI/ML documents and practiced cosine similarity search and metadata filtering. I also learned how documents can be split into smaller chunks and stored as vectors for retrieval. Finally, I combined ChromaDB with Ollama to build a simple retrieval-augmented generation workflow, where the top-3 relevant PDF chunks were retrieved and passed to the local LLM to generate a context-grounded answer.



## W5D4 - Semantic Search with ChromaDB

### Task Completed

Implemented a separate ChromaDB semantic search workflow with 20 AI/ML documents, cosine similarity search, metadata filtering, PDF chunk retrieval, and Ollama-based RAG.

### Checklist

- [x] Installed and verified ChromaDB.
- [x] Created a separate W5D4 ChromaDB collection.
- [x] Added 20 AI/ML documents with embeddings.
- [x] Configured cosine similarity.
- [x] Performed similarity search.
- [x] Performed metadata filtering.
- [x] Manually verified search and filtering results.
- [x] Created a separate W5D4 AI/ML reference PDF.
- [x] Extracted PDF text using pypdf.
- [x] Split the PDF into chunks.
- [x] Stored PDF chunks in ChromaDB.
- [x] Retrieved the top-3 PDF chunks.
- [x] Passed retrieved chunks to Ollama.
- [x] Verified the Ollama answer against the retrieved context.
- [x] Generated W5D4 output evidence.
- [x] Created manual verification evidence.
- [x] Tested the W5D4 implementation successfully.
- [x] Created the first W5D4 implementation commit.
- [ ] Created the second descriptive W5D4 self-review commit.
- [ ] Pushed the W5D4 changes to GitHub.
- [ ] Raised/updated the Pull Request.

### Evidence Generated

- output_evidence/w5d4/chromadb_setup_results.txt
- output_evidence/w5d4/pdf_rag_results.txt
- output_evidence/w5d4/manual_verification.md
- data/w5d4/ai_ml_reference.pdf

### Reflection

Today I learned how semantic search can retrieve relevant information based on meaning rather than exact keyword matching. I created a separate W5D4 ChromaDB collection containing 20 AI/ML documents and practiced cosine similarity search and metadata filtering. I also learned how PDF content can be divided into chunks and retrieved using vector similarity. Finally, I combined ChromaDB with Ollama to retrieve the top-3 relevant chunks and generate a context-based answer.

## W5D5 - Week 5 Project: Local Q&A Bot

### Task Completed

Implemented a local Q&A workflow using Ollama and compared llama3.2:3b with qwen2.5:3b using the same three AI/ML questions.

### Checklist

- [x] Verified Ollama installation.
- [x] Verified llama3.2:3b is available locally.
- [x] Verified qwen2.5:3b is available locally.
- [x] Created a Python script for Ollama API calls.
- [x] Added a custom system prompt.
- [x] Tested 5 prompts successfully.
- [x] Compared llama3.2:3b and qwen2.5:3b on the same 3 questions.
- [x] Manually reviewed response differences.
- [x] Documented model comparison results.
- [x] Generated W5D5 output evidence.
- [x] Tested the W5D5 implementation successfully.
- [x] Created the first W5D5 implementation commit.
- [ ] Created the second W5D5 self-review commit.
- [ ] Pushed the W5D5 changes to GitHub.
- [ ] Updated the Pull Request.

### Evidence Generated

- output_evidence/w5d5/qa_bot_results.txt
- output_evidence/w5d5/model_comparison_results.txt
- output_evidence/w5d5/model_comparison.md

### Reflection

Today I learned how to interact with local large language models through the Ollama API. I created a Python-based local Q&A bot with a custom system prompt and tested five AI/ML questions. I also compared llama3.2:3b and qwen2.5:3b using the same three questions and documented differences in response style, examples, structure, and level of detail.

## W6D1 - LangChain Fundamentals: Chains & Prompts

### Task Completed

Implemented and tested LangChain fundamentals using an Ollama LLM, including a prompt chain, conversation memory, and a simple agent with two tools.

### Checklist

- [x] Installed and verified LangChain packages.
- [x] Verified Ollama models are available locally.
- [x] Created a PromptTemplate -> Ollama LLM -> StrOutputParser chain.
- [x] Tested the chain with 5 inputs.
- [x] Added ConversationBufferMemory.
- [x] Tested conversation memory across 5 turns.
- [x] Verified that 10 messages were stored in conversation history.
- [x] Created a simple LangChain agent.
- [x] Added a web search stub tool.
- [x] Added a calculator tool.
- [x] Tested the agent with 3 tasks.
- [x] Verified all 3 agent tasks completed successfully.
- [x] Generated W6D1 output evidence.
- [x] Tested the W6D1 implementation successfully.
- [x] Created the first W6D1 implementation commit.
- [ ] Created the second W6D1 self-review commit.
- [ ] Pushed the W6D1 changes to GitHub.
- [ ] Raised the Pull Request.

### Evidence Generated

- output_evidence/w6d1/langchain_fundamentals_results.txt

### Reflection

Today I learned the fundamentals of LangChain chains, prompts, conversation memory, and agents. I created a PromptTemplate to Ollama LLM to StrOutputParser workflow and tested it with five inputs. I also used ConversationBufferMemory to maintain conversation history across five turns. Finally, I created a simple agent with a web search stub and calculator tool and verified it with three tasks.



## W6D2 - LangChain Memory & Conversation History

### Task Completed

Implemented and tested LangChain memory and conversation history using a local Ollama LLM. The implementation includes a LangChain prompt chain, ConversationBufferMemory, and a two-tool LangChain agent.

### Checklist

- [x] Implemented the LangChain prompt chain.
- [x] Tested the chain with 5 inputs.
- [x] Added ConversationBufferMemory.
- [x] Tested conversation memory across 5 turns.
- [x] Verified that 10 messages were stored in conversation history.
- [x] Created a two-tool LangChain agent.
- [x] Added a web search stub tool.
- [x] Added a calculator tool.
- [x] Tested the agent with 3 tasks.
- [x] Verified all 3 agent tasks completed successfully.
- [x] Generated W6D2 output evidence.
- [x] Tested the W6D2 implementation successfully.
- [x] Created the first W6D2 implementation commit.
- [x] Created the second W6D2 self-review commit.
- [ ] Pushed the W6D2 changes to GitHub.
- [ ] Raised the Pull Request.

### Evidence Generated

- output_evidence/w6d2/langchain_memory_agent_results.txt

### Reflection

Today I learned how LangChain can be used to build chains, maintain conversation history, and create tool-using agents. I tested a prompt chain with five inputs, maintained conversation history across five turns using ConversationBufferMemory, and verified that ten messages were stored. I also created a two-tool agent with a web search stub and calculator tool and successfully tested three tasks.

## W6D3 - LangChain Tools & Agents

### Task Completed

Implemented and tested LangChain tools and agents using a local Ollama LLM. The implementation includes a prompt chain, ConversationBufferMemory, and a two-tool LangChain agent.

### Checklist

- [x] Implemented the LangChain PromptTemplate -> Ollama LLM -> StrOutputParser chain.
- [x] Tested the chain with 5 inputs.
- [x] Added ConversationBufferMemory.
- [x] Tested conversation memory across 5 turns.
- [x] Verified that 10 messages were stored in conversation history.
- [x] Created a two-tool LangChain agent.
- [x] Added a web search stub tool.
- [x] Added a calculator tool.
- [x] Tested the agent with 3 tasks.
- [x] Verified all 3 agent tasks completed successfully.
- [x] Generated W6D3 output evidence.
- [x] Tested the W6D3 implementation successfully.
- [x] Created the first W6D3 implementation commit.
- [x] Created the second W6D3 self-review commit.
- [ ] Pushed the W6D3 changes to GitHub.
- [x] Existing Week 6 Pull Request #8 will be updated with W6D3 changes.

### Evidence Generated

- output_evidence/w6d3/langchain_tools_agents_results.txt

### Reflection

Today I learned how LangChain can be used to build prompt chains, maintain conversation history, and create tool-using agents. I tested the prompt chain with five inputs, maintained conversation history across five turns using ConversationBufferMemory, and verified that ten messages were stored. I also created a two-tool agent with a web search stub and calculator tool and successfully tested three tasks.

## W6D4 - RAG Pipeline - LangChain + ChromaDB

### Task Completed

Implemented and tested a RAG pipeline using ChromaDB, embeddings, similarity search, PDF retrieval, and a local Ollama LLM.

### Checklist

- [x] Verified ChromaDB installation.
- [x] Created a ChromaDB collection using cosine distance.
- [x] Added 20 documents with embeddings.
- [x] Performed cosine similarity search.
- [x] Performed metadata filtering.
- [x] Manually verified similarity search and metadata filtering results.
- [x] Loaded the AI/ML reference PDF.
- [x] Split the PDF into 3 chunks.
- [x] Stored PDF chunks in ChromaDB with embeddings and metadata.
- [x] Retrieved the top 3 PDF chunks.
- [x] Passed retrieved context to Ollama llama3.2:3b.
- [x] Verified the generated answer.
- [x] Generated W6D4 output evidence.
- [x] Tested the W6D4 implementation successfully.
- [ ] Created the first W6D4 implementation commit.
- [ ] Created the second W6D4 self-review commit.
- [ ] Pushed the W6D4 changes to GitHub.
- [x] Existing Week 6 Pull Request #8 will be updated with W6D4 changes.

### Evidence Generated

- output_evidence/w6d4/chromadb_rag_results.txt

### Reflection

Today I learned how ChromaDB can be used as a vector store for embeddings, similarity search, and metadata filtering. I added 20 machine learning documents and verified cosine similarity retrieval. I also built a PDF RAG pipeline by splitting the reference PDF into three chunks, storing the chunks in ChromaDB, retrieving the top three relevant chunks, and passing the retrieved context to the local Ollama llama3.2:3b model to generate an answer.

## W6D5 - Week 6 Project - Document Chatbot with LangChain

### Task Completed

Implemented and tested a LangChain document chatbot workflow using a PromptTemplate, local Ollama LLM, output parser, conversation memory, and a two-tool agent.

### Checklist

- [x] Implemented the LangChain PromptTemplate -> Ollama LLM -> StrOutputParser chain.
- [x] Tested the chain with 5 inputs.
- [x] Added ConversationBufferMemory.
- [x] Tested conversation memory across 5 turns.
- [x] Verified that 10 messages were stored in conversation history.
- [x] Created a two-tool LangChain agent.
- [x] Added a web search stub tool.
- [x] Added a calculator tool.
- [x] Tested the agent with 3 tasks.
- [x] Verified all 3 agent tasks completed successfully.
- [x] Generated W6D5 output evidence.
- [x] Tested the W6D5 implementation successfully.
- [x] Created the first W6D5 implementation commit.
- [x] Created the second W6D5 self-review commit.
- [ ] Pushed the W6D5 changes to GitHub.
- [x] Existing Week 6 Pull Request #8 will be updated with W6D5 changes.

### Evidence Generated

- output_evidence/w6d5/langchain_document_chatbot_results.txt

### Reflection

Today I learned how LangChain can be used to build prompt chains, maintain conversation history, and create tool-using agents. I tested the prompt chain with five inputs, maintained conversation history across five turns using ConversationBufferMemory, and verified that ten messages were stored. I also created a two-tool agent with a web search stub and calculator tool and successfully tested three tasks.
## W7D1: Haystack Pipeline Architecture

### Tasks Completed

* [x] Built a Haystack retrieval pipeline using the current Haystack API.
* [x] Loaded and indexed 5 PDF documents.
* [x] Created a BM25 retriever and executed 10 evaluation questions.
* [x] Evaluated BM25 retrieval quality using Precision@1.
* [x] Replaced BM25 retrieval with dense embedding retrieval.
* [x] Used `sentence-transformers/all-MiniLM-L6-v2` for dense embeddings.
* [x] Compared BM25 and dense retrieval on the same 10 questions.
* [x] Saved retrieval results and evaluation evidence in `output_evidence/w7d1/`.

### Results

* Number of PDF documents: 5
* Number of evaluation questions: 10
* BM25 Precision@1: 100.00%
* Dense Retrieval Precision@1: 100.00%
* Difference: 0.00%

### Implementation Note

The project uses the current Haystack API available in the environment. `PyPDFToDocument` is used for PDF conversion, `InMemoryDocumentStore` is used for document storage, and `InMemoryBM25Retriever` / `InMemoryEmbeddingRetriever` are used for retrieval. The older Reader component referenced in the assignment is not exposed by the installed Haystack version, so the implementation uses the currently supported retrieval components.

### Evidence

* `w7d1_haystack_pipeline.py`
* `create_w7d1_pdfs.py`
* `w7d1_data/` containing 5 PDF documents
* `output_evidence/w7d1/haystack_retrieval_results.txt`

### Completion

W7D1 practical tasks were completed and tested successfully. Both BM25 and dense retrieval achieved 100% Precision@1 on the 10-question evaluation set.


## W7D2: Haystack Retrieval — BM25 & Dense Retrieval

### Tasks Completed

* [x] Built a Haystack retrieval pipeline using the current Haystack API.
* [x] Created and indexed 5 PDF documents.
* [x] Created a BM25 retriever and tested 10 evaluation questions.
* [x] Created dense embeddings using sentence-transformers/all-MiniLM-L6-v2.
* [x] Created a dense embedding retriever and tested the same 10 questions.
* [x] Compared BM25 and dense retrieval using Precision@1.
* [x] Manually evaluated the top-1 retrieved document for all 10 questions.
* [x] Generated W7D2 retrieval output evidence.
* [x] Tested the W7D2 implementation successfully.

### Results

* Number of PDF documents: 5
* Number of evaluation questions: 10
* BM25 Precision@1: 100.00%
* Dense Retrieval Precision@1: 100.00%
* Difference: 0.00%
* BM25 correct results: 10/10
* Dense correct results: 10/10

### Implementation Note

The project uses the current Haystack API available in the environment. PyPDFToDocument is used for PDF conversion, InMemoryDocumentStore is used for document storage, and InMemoryBM25Retriever / InMemoryEmbeddingRetriever are used for retrieval. The older Reader component referenced in the assignment is not exposed by the installed Haystack version, so the implementation uses the currently supported retrieval components.

### Evidence

* w7d2_haystack_retrieval.py
* create_w7d2_pdfs.py
* w7d2_data/ containing 5 PDF documents
* output_evidence/w7d2/haystack_retrieval_results.txt

### Completion

W7D2 practical tasks were completed and tested successfully. Both BM25 and dense retrieval achieved 100% Precision@1 on the same 10-question evaluation set. All 20 retrieval evaluations were marked CORRECT.

## W7D3: LlamaIndex — Document Indexing & RAG

### Tasks Completed

* [x] Created and indexed 5 text documents using LlamaIndex `VectorStoreIndex`.
* [x] Configured Ollama embeddings using `nomic-embed-text`.
* [x] Built a LlamaIndex `QueryEngine`.
* [x] Ran 10 queries against the indexed documents.
* [x] Verified the retrieved source document for all 10 queries.
* [x] Connected LlamaIndex to ChromaDB as the vector store.
* [x] Re-ran the same 10 queries using the ChromaDB-backed index.
* [x] Compared query latency between the default vector store and ChromaDB.
* [x] Saved W7D3 results and evaluation evidence.

### Results

* Number of text documents: 5
* Number of queries: 10
* Default vector store source verification: 10/10
* ChromaDB source verification: 10/10
* Default average latency: 4715.50 ms
* ChromaDB average latency: 3610.00 ms
* Measured latency difference: 1105.50 ms lower with ChromaDB in this run

### Implementation Note

The implementation uses `llama3.2:3b` as the Ollama language model and `nomic-embed-text` for embeddings. The LlamaIndex query configuration uses a 2048-token context window and `similarity_top_k=1` to keep local model memory usage manageable.

### Evidence

* `w7d3_llamaindex_rag.py`
* `w7d3_data/` containing 5 text documents
* `output_evidence/w7d3/llamaindex_rag_results.txt`

### Completion

W7D3 practical tasks were completed and tested successfully. All 10 queries were source-verified with the default LlamaIndex vector store, and the same 10 queries were source-verified again using ChromaDB. Latency comparison evidence was saved successfully.

## W7D4: LlamaIndex + Ollama - Local RAG

### Tasks Completed

* [x] Verified Ollama installation and local Ollama API availability.
* [x] Verified llama3.2:3b and qwen2.5:3b models were available locally.
* [x] Ran the first local inference using llama3.2:3b.
* [x] Built a Python script to call the Ollama local API.
* [x] Added a custom system prompt for AI/ML explanations.
* [x] Tested llama3.2:3b with 5 prompts.
* [x] Compared llama3.2:3b and qwen2.5:3b using the same 3 questions.
* [x] Documented differences in response style, structure, detail, and factual accuracy.
* [x] Saved W7D4 inference and model comparison evidence.

### Results

* Ollama version: 0.34.2
* Primary model: llama3.2:3b
* Comparison model: qwen2.5:3b
* Custom system prompt: AI/ML learning assistant focused on clear and concise explanations
* Number of Task 2 prompts: 5
* Number of model comparison questions: 3
* Local Ollama API calls completed successfully: 11

### Model Comparison Observations

* llama3.2:3b generally produced longer, example-oriented explanations.
* qwen2.5:3b generally produced more concise and structured explanations.
* Both models gave broadly relevant answers to the machine-learning question.
* For the RAG question, qwen2.5:3b correctly described Retrieval-Augmented Generation, while the observed llama3.2:3b response incorrectly interpreted RAG as a different concept.
* Both models provided relevant advantages and use cases for vector databases.
* The comparison demonstrates that locally generated responses should be verified against reliable source material, particularly for technical concepts.

### Evidence

* w7d4_ollama_local_inference.py
* output_evidence/w7d4/ollama_local_inference_results.txt
* output_evidence/w7d4/model_comparison.md

### Completion

W7D4 practical tasks were completed and tested successfully. Ollama local inference was verified, the custom API script successfully completed 5 prompt tests, and both local models were compared using the same 3 questions. Response differences were documented and evidence was saved successfully.

## W7D5: Multi-document RAG System

### Tasks Completed

* [x] Created a separate five-document knowledge corpus in `w7d5_data/`.
* [x] Implemented a multi-document RAG pipeline using LangGraph.
* [x] Implemented document retrieval and answer generation nodes.
* [x] Connected the pipeline to the local Ollama model `llama3.2:3b`.
* [x] Ran five questions against the document collection.
* [x] Verified the expected source documents for all five questions.
* [x] Achieved a source verification rate of 5/5, or 100%.
* [x] Integrated CrewAI using an agent, task, and crew.
* [x] Imported and verified Ragas version 0.4.3.
* [x] Performed a Ragas integration and source-verification check.
* [x] Logged the experiment and metrics using MLflow with a SQLite backend.
* [x] Saved TXT and JSON execution evidence.
* [x] Added generated MLflow and ChromaDB artifacts to `.gitignore`.
* [x] Executed and tested the complete W7D5 pipeline successfully.

### Results

* Documents loaded: 5
* RAG queries executed: 5
* Verified source matches: 5/5
* Source match rate: 100%
* Ragas version: 0.4.3
* MLflow experiment: `W7D5_Multi_Document_RAG`
* MLflow run ID: `0c661e25dd414c6eabdbdc16db1be451`

### Observations

* LangGraph successfully connected retrieval and generation into a workflow.
* Ollama generated answers using the retrieved document context.
* CrewAI successfully executed the retrieval and generation explanation task.
* Model-generated content should still be reviewed for factual accuracy.
* The Ragas work in this task was an integration and source-verification check, not a full Ragas metric evaluation.
* The generated MLflow database was excluded from Git because it is a runtime artifact.

### Evidence

* `w7d5_multi_document_rag.py`
* `w7d5_data/`
* `output_evidence/w7d5/multi_document_rag_results.txt`
* `output_evidence/w7d5/multi_document_rag_results.json`

### Completion

W7D5 multi-document RAG implementation, integration checks, execution, and evidence collection were completed successfully.

# W8D2 Self Review - Ragas Evaluation

## Objective

Evaluate the W7D5 multi-document RAG pipeline using Ragas and compare a baseline retrieval configuration with an optimization candidate.

## Implementation

* Evaluated 10 RAG question-answer samples.
* Used 5 W7D5 source documents.
* LLM: `llama3.2:3b` via Ollama.
* Embeddings: `nomic-embed-text:latest`.
* Ragas metrics:

  * Faithfulness
  * Answer Relevancy
  * Context Precision
  * Context Recall
* Baseline retrieval: `top_k=2`.
* Optimization candidate: `top_k=3`.

## Results

| Metric            | Baseline top_k=2 | Optimized top_k=3 |
| ----------------- | ---------------: | ----------------: |
| Faithfulness      |           0.9024 |            0.9250 |
| Answer Relevancy  |           0.8883 |            0.8934 |
| Context Precision |           1.0000 |            1.0000 |
| Context Recall    |           0.8450 |            0.9057 |
| Mean              |           0.9089 |            0.9310 |

## Outcome

The `top_k=3` configuration produced a higher mean evaluation score than the `top_k=2` baseline and was selected by the evaluation script.

## Evidence

* `output_evidence/w8d2/qa_pairs.json`
* `output_evidence/w8d2/ragas_evaluation_results.json`
* `output_evidence/w8d2/ragas_evaluation_results.txt`

## Validation

* Python compilation succeeded.
* All 5 W7D5 documents loaded successfully.
* Structured LLM output using JSON schema initialized successfully.
* All 40 baseline metric evaluations completed.
* All 40 optimized metric evaluations completed.
* Valid numeric Ragas scores were produced for all four metrics.

## Conclusion

W8D2 successfully demonstrated Ragas-based evaluation of the multi-document RAG pipeline and an experiment comparing retrieval configurations.



