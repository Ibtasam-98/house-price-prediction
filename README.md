# Housing Price Category Prediction Project

This project implements a machine learning pipeline to predict whether a house falls into a high price category based on various features. The pipeline includes data preprocessing, visualization, model training, and prediction functionalities.

## Overview

The project consists of the following main components:

-   **`main.py`**: The main script that orchestrates the entire pipeline, from data loading and preprocessing to model training, evaluation, and prediction.
-   **`data_preprocessing.py`**: Handles loading the dataset, encoding categorical features, scaling numerical features, and creating the binary target variable based on the median price. It also includes outlier handling using the Z-score method.
-   **`data_visualization.py`**: Generates various visualizations to understand the dataset, including correlation heatmaps, price distributions, and relationships between features and price.
-   **`model_training.py`**: Trains a Random Forest Classifier on the preprocessed data, evaluates its performance using metrics like accuracy, classification report, confusion matrix, MSE, and R2 score, and saves visualizations of the evaluation results.
-   **`prediction.py`**: Provides an interactive command-line interface for users to input house features and get a prediction on the price category (Expensive or Not Expensive) using the trained model.

## Setup

To run this project you need to have Python 3 installed on your system along with the following libraries:

-   pandas
-   scikit-learn
-   matplotlib
-   seaborn
-   numpy

You can install these libraries using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
