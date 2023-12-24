# Classification Models Diabetes Prediction
![My Skills](https://skillicons.dev/icons?i=python)

## Introduction
This project is aimed at predicting the onset of diabetes using a dataset that includes various health indicators from female patients of Pima Indian heritage. By employing four different classification models—K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), and Decision Trees—this endeavor seeks to enhance preventive care and facilitate timely healthcare interventions.

## Authors
Michael Ginzburg

## Date
December 9, 2023

## Overview
The project begins with an exploratory data analysis (EDA) to understand the relationships within the health data. The classification models were selected for their unique approaches to handling binary classification tasks, as outlined in resources from the Global Tech Council and Analytics Vidhya. The dataset used is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases and is available on Kaggle.

## Data
The dataset comprises 768 rows and 9 columns, using eight variables to predict the outcome of diabetes. It is exclusively composed of data from women and includes variables such as Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.

## Libraries and Tools Used
Python 3 and Jupyter Notebook environment (Google Colab)
numpy for numerical operations
pandas for data manipulation and analysis
matplotlib.pyplot and seaborn for data visualization
sklearn for machine learning models, data splitting, model evaluation, and metrics
impyute for imputation of missing values
mlxtend for plotting decision regions
Exploratory Data Analysis
The data analysis includes identifying missing values represented as zeroes, plotting distributions, and visualizing relationships between variables with scatter plots and pair plots.

## Modeling
Logistic Regression
The Logistic Regression model acts as a baseline for performance comparison, evaluated using metrics such as accuracy, precision, recall, and the ROC curve.

## Decision Trees
Decision Trees offer insights into decision-making with their interpretable structure. The model is optimized by experimenting with tree depth and leaf nodes.

## SVM
The SVM model's effectiveness in high-dimensional spaces is assessed through confusion matrices, F1 scores, and accuracy metrics.

## KNN
KNN predictions are based on the proximity of data points, with the optimal number of neighbors determined by analyzing ROC accuracy across different values.

## Model Evaluation
The models are evaluated on their ability to accurately predict diabetes, using various metrics like the confusion matrix, classification report, ROC curve, and AUC score.

## Conclusion
The project concludes with the Decision Tree model demonstrating the most promising results with an accuracy score of 87%. This interpretable model is identified as a valuable tool for healthcare professionals in early diabetes detection and intervention. The report acknowledges potential improvements through ensemble methods and further experimentation with feature engineering and class imbalance.

## Setup and Execution
Data is loaded and preprocessed, including handling missing values and normalizing skewed distributions.
Models are trained on the dataset, and their performance is evaluated.
The best-performing model is identified and its suitability for deployment in a healthcare setting is discussed.
Future Work
The analysis suggests potential improvements through ensemble methods, feature engineering, and addressing class imbalance to enhance model performance further.
