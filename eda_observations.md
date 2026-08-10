# W1D4 EDA Observations

## Five Key Observations

1. The dataset contains 108 rows and 94 columns. It includes population, household, literacy, social category, and workforce-related information.

2. The dataset contains 91 numeric columns and 3 categorical columns. The numeric columns contain population and workforce counts, while columns such as Level, Name, and TRU contain categorical information.

3. There are no missing values in the dataset. The `df.isnull().sum()` output shows zero missing values across all 94 columns.

4. Most numeric variables are strongly right-skewed. The distribution plots show that many observations have relatively small values, while a smaller number of observations have much larger population or workforce counts.

5. Several population and workforce variables have strong positive correlations. This is expected because variables such as total population, male population, female population, and workforce counts are related to the size of the population.

## EDA Narrative

The dataset contains 108 records and 94 columns describing population, household, literacy, social categories, and workforce characteristics. The dataset is relatively small in terms of rows but contains many numerical variables. The descriptive statistics show large differences between the minimum and maximum values of several population-related variables. The distribution plots indicate that many numerical variables are right-skewed, with most observations concentrated at lower values and a smaller number of observations having much larger values. This is reasonable because population sizes can vary considerably between regions.

The missing-value analysis shows that there are no missing values across the dataset, which reduces the need for missing-value imputation. The correlation heatmap shows strong positive relationships between several population and workforce variables. These relationships are expected because male and female population counts contribute to total population, while workforce measures are also related to population size.

One suspicious characteristic is the presence of very large differences in scale between variables. Some variables contain values in the hundreds or thousands, while others reach millions. This should be considered before applying machine learning algorithms. Overall, the dataset is clean but may require scaling, feature selection, and handling of highly skewed variables before modelling.