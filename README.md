# Airbnb Listings Classification with Logistic Regression

This project demonstrates the full evaluation phase of the machine learning lifecycle using **Logistic Regression** to solve a classification problem. We used **Python** on **Jupyter Notebook** along with libraries such as **NumPy**, **Pandas**, and **scikit-learn** to analyze and predict on Airbnb listings data.

## Project Overview

The objective of this project is to predict a specific label for Airbnb listings based on selected features. Through this lab, I applied logistic regression and fine-tuned the model with a **grid search** to find the optimal hyperparameters, compared model performance using evaluation metrics, and made the model persistent for future use.

### Key Tasks

1. **Data Preparation**:  
   - **Load the Dataset**: Loaded the Airbnb listings dataset into a Pandas DataFrame.
   - **Define the Label and Features**: Identified the target variable (label) for classification and the features used for prediction.
   - **Create Labeled Examples**: Preprocessed the data to create labeled examples for training.

2. **Data Splitting**:
   - Split the data into **training** and **testing** sets to evaluate model performance effectively.

3. **Model Training and Testing**:
   - **Initial Logistic Regression Model**: Trained a logistic regression model using scikit-learn with default hyperparameters.
   - **Hyperparameter Tuning (Grid Search)**: Performed a grid search to identify the optimal hyperparameters for logistic regression. This included testing different regularization values to improve model accuracy and prevent overfitting.
   - **Evaluation of Optimized Model**: Trained, tested, and evaluated a logistic regression model using the optimal hyperparameters obtained from the grid search.

4. **Model Evaluation**:
   - **Precision-Recall Curve**: Plotted precision-recall curves for both the default and optimized models to visualize the trade-off between precision and recall.
   - **ROC Curve and AUC**: Computed and plotted the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) for both models to assess performance.

5. **Feature Selection**:
   - Conducted feature selection to identify the most influential variables, improving the model’s interpretability and potentially reducing complexity.

6. **Model Persistence**:
   - Saved the trained model, making it persistent and ready for future predictions without retraining.

## Tech Stack

- **Programming Language**: Python
- **Development Environment**: Jupyter Notebook
- **Libraries**: 
  - **NumPy**: For numerical operations and data manipulation.
  - **Pandas**: For data loading, preprocessing, and DataFrame management.
  - **scikit-learn**: For implementing logistic regression, hyperparameter tuning, and model evaluation.

## Results

- **Model Comparison**: Compared the initial model with the optimized model using precision-recall and ROC-AUC metrics to evaluate improvements from hyperparameter tuning.
- **Optimal Model Performance**: Achieved higher precision, recall, and AUC with the optimized model, demonstrating the effectiveness of grid search in improving logistic regression model accuracy.

## What I Learned

This project provided hands-on experience with:
- **Model Selection and Evaluation**: Learned to choose and evaluate models effectively using logistic regression for classification problems.
- **Hyperparameter Tuning**: Applied grid search to find the optimal regularization parameter, enhancing model performance.
- **Model Persistence**: Saved the model for future use, essential for creating reusable machine learning solutions.


