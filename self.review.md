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
- [ ] Created the second required commit.
- [ ] Pushed changes to GitHub.
- [ ] Raised/updated the Pull Request.

### W2D2 Top 5 Features

- **TOT_M:** Total male population is a major component of total population.
- **TOT_F:** Total female population is a major component of total population.
- **NON_WORK_P:** Represents the non-working population and provides information about population composition.
- **TOT_WORK_P:** Represents the total working population and provides information about the working population.
- **P_LIT:** Represents the literate population and provides information about literacy within the population.

### Reflection

Today I learned how feature scaling and feature selection can prepare numerical data for machine learning. I applied StandardScaler, MinMaxScaler, and RobustScaler to numeric predictor features and compared their distributions before and after scaling. I also used SelectKBest with f_regression to identify the five features with the strongest linear relationships with the numeric target TOT_P. CIA review helped me identify that the target variable should not be used for the scaling demonstration and guided me to apply the scalers to predictor features instead. I tested the corrected implementation successfully and verified the generated output evidence.