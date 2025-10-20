# AI_ML_Task1: Titanic Dataset Preprocessing 

## Project Overview

In this project, I cleaned and prepared the Titanic dataset for machine learning using **Python**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn** in **Google Colab**.  

The main objective was to preprocess the dataset to make it suitable for predictive modeling while ensuring data quality and consistency.

## Data Cleaning and Preparation
1. **Data Exploration:**  
   - Loaded the dataset and examined its structure, data types, missing values, and summary statistics.  
   - Identified columns requiring cleaning, such as Age, Embarked, and Cabin.

2. **Handling Missing Values:**  
   - Filled missing values in the **Age** column with the median.  
   - Filled missing values in the **Embarked** column with the most frequent value.  
   - Dropped the **Cabin** column due to excessive missing data.

3. **Encoding Categorical Features:**  
   - Converted the **Sex** column to numeric (male=0, female=1).  
   - Applied one-hot encoding to the **Embarked** column for machine-learning compatibility.

4. **Feature Scaling:**  
   - Standardized numerical features (**Age** and **Fare**) to normalize the scale.

5. **Outlier Detection and Removal:**  
   - Visualized outliers using **boxplots**.  
   - Removed extreme values using the **IQR (Interquartile Range) method** for data consistency.

6. **Final Dataset:**  
   - The dataset is fully cleaned, encoded, and normalized.  
   - Ready for building predictive models such as logistic regression, decision trees, or ensemble methods.

## Files in this Repository
- `task1.ipynb` – Jupyter notebook with all preprocessing steps.  
- `Titanic-Dataset.csv` – Original dataset.  
- `cleaned_titanic.csv` – Cleaned and preprocessed dataset.  
- `README.md` – This file.

---

This README provides a summary of the **data cleaning and preprocessing workflow**, highlighting the techniques used to make the Titanic dataset machine-learning ready.
