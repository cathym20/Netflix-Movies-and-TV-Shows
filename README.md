# Movie vs TV Show Classification

## Project Overview

This project focuses on building and comparing machine learning models to classify streaming content as either a Movie or a TV Show using metadata and textual descriptions. Rather than aiming only for high accuracy, the main objective was to understand how different models learn from data, how individual features influence predictions, and how model complexity affects performance.

## Dataset

The dataset consists of over 5,300 entries describing movies and TV shows. It includes information such as title, director, cast, country, release year, rating, duration, genre listings, and a short text description. The duration column was processed into a numeric format to make it usable for modelling. Basic data cleaning and preprocessing were performed before training the models.

## Problem Statement

The task is to predict whether a given piece of content is a Movie or a TV Show based on the available features. This is framed as a binary classification problem.

## Logistic Regression Model

Logistic Regression was used as the baseline model in order to clearly understand how linear classification works. The text descriptions were converted into numerical features using TF-IDF vectorization, and numeric features were scaled where required. Different values of the regularization parameter were tested to observe how model complexity and coefficient magnitude changed.

The model achieved near-perfect performance across a wide range of regularization strengths. Further inspection showed that the duration feature had a very strong influence on predictions, while most text features were assigned zero or near-zero weights when L1 regularization was used. This indicated that only a small subset of words was needed to separate the classes.

## Random Forest Model

A Random Forest classifier was trained using the same train–test split to allow a fair comparison with Logistic Regression. This model was chosen to introduce non-linearity and to observe how an ensemble method handles the same features.

The Random Forest model produced similar performance to Logistic Regression. Feature importance analysis showed that the duration feature alone accounted for a large proportion of the total importance, confirming that it was a dominant signal regardless of model choice. Increasing model complexity did not result in meaningful performance gains.

## Observations

The results show that when a dataset contains very strong predictive features, even simple models can perform extremely well. In such cases, more complex models do not necessarily provide better results. Regularization becomes important only when the model is forced to rely on weaker or noisier signals. Across both models, duration emerged as the most influential feature, while textual features played a secondary role.

## Libraries

This project was implemented in Python using Pandas and NumPy for data manipulation, Scikit-learn for modelling and evaluation, and Matplotlib for basic visualisation during exploratory analysis.

## Conclusion

This project demonstrates a complete machine learning workflow, from preprocessing and feature engineering to model training, evaluation, and interpretation. It emphasizes understanding model behaviour and feature influence rather than focusing solely on performance metrics.


