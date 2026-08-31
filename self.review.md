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
- [x] Evaluated the model using R², MAE, and RMSE.
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
- [x] Evaluated the LinearRegression model using R².
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

Today I learned how to build and evaluate regression models using Scikit-Learn. I trained LinearRegression on a real dataset and evaluated it using MSE, RMSE, MAE, and R². I also learned how predicted-vs-actual and residual plots help evaluate regression performance. Finally, I compared LinearRegression with Ridge and Lasso regression and documented the results as output evidence. CIA review helped me verify the implementation and improve the overall quality of the W3D1 work.
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