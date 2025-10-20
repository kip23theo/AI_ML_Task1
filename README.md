# AI_ML_Task1: Titanic Dataset Preprocessing 🛳️

This repository contains the preprocessing workflow for the Titanic dataset using Google Colab.  
The main goal is to clean, encode, normalize, and optionally remove outliers, preparing the data for analysis or machine learning tasks. Proper preprocessing is essential because real-world datasets often contain missing values, categorical variables, and inconsistencies that can negatively impact model performance.  

## Files in this Repository
- `task1.ipynb` – Jupyter notebook containing all preprocessing steps.  
- `Titanic-Dataset.csv` – Original Titanic dataset.  
- `cleaned_titanic.csv` – Cleaned and preprocessed dataset.  
- `README.md` – This file.  

## Preprocessing Steps and Theory

### Step 1: Open Google Colab
Google Colab is a cloud-based environment that allows you to run Python code in a Jupyter notebook interface without installing anything locally. It supports libraries such as pandas, NumPy, and scikit-learn, which are essential for data preprocessing. A new notebook was created and renamed `task1.ipynb` for this project.

### Step 2: Upload Dataset
The dataset file `Titanic-Dataset.csv` was uploaded to the Colab environment. Uploading files allows the notebook to access and manipulate the data in memory. This step is necessary because Colab uses temporary virtual storage.


from google.colab import files
uploaded = files.upload()


Step 3: Load the Dataset

Using pandas, the dataset was loaded into a DataFrame. Pandas provides tools to inspect, manipulate, and clean tabular data efficiently.

import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")
df.head()


Step 4: Explore the Data

Exploratory data analysis (EDA) is critical to understand the dataset. We inspected data types, missing values, and basic statistics. This helps identify potential issues, such as missing entries or anomalies, before applying machine learning algorithms.

df.info()
df.describe()
df.isnull().sum()


Step 5: Handle Missing Values

Real-world datasets often contain missing data. In this step:

Missing Age values were replaced with the median, which is robust against outliers.

Missing Embarked values were filled with the mode (most frequent value).

The Cabin column was dropped because it had too many missing values, which could bias the model.

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.isnull().sum()

Step 6: Encode Categorical Features

Machine learning algorithms generally require numerical input. Categorical features such as Sex and Embarked were converted into numeric representations:

Sex was mapped to 0 (male) and 1 (female).

Embarked was one-hot encoded, creating binary columns for each category.

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.head()

Step 7: Normalize Numeric Columns

Normalization or standardization ensures that features like Age and Fare are on a similar scale. This prevents features with larger ranges from dominating the learning process, especially in algorithms like gradient descent-based models.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
df[['Age', 'Fare']].head()

Step 8: Visualize Outliers

Outliers can skew statistical measures and negatively impact model performance. Boxplots were used to visually inspect the distributions of Age and Fare.

import matplotlib.pyplot as plt
import seaborn as sns
for col in ['Age', 'Fare']:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

Step 9: Remove Outliers (Optional)

Outliers were optionally removed using the Interquartile Range (IQR) method. This method defines outliers as values beyond 1.5 times the IQR below Q1 or above Q3. Removing outliers can improve model performance by reducing noise.

for col in ['Age', 'Fare']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
df.shape

Step 10: Save the Cleaned Dataset

The cleaned dataset was saved as cleaned_titanic.csv and downloaded for further analysis. Saving the preprocessed dataset ensures that the same data can be used consistently across different models.

df.to_csv("cleaned_titanic.csv", index=False)
from google.colab import files
files.download("cleaned_titanic.csv")

Outcome

The Titanic dataset has been successfully cleaned, encoded, standardized, and optionally filtered for outliers.
This preprocessed dataset is now ready for exploratory data analysis or input into machine learning models such as logistic regression, decision trees, or neural networks.
