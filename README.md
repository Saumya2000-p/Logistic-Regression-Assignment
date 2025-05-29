# Logistic-Regression-Assignment

### Theoretical Questions

1. What is Logistic Regression, and how does it differ from Linear Regression?
2. What is the mathematical equation of Logistic Regression?
3. Why do we use the Sigmoid function in Logistic Regression?
4. What is the cost function of Logistic Regression?
5. What is Regularization in Logistic Regression? Why is it needed?
6. Explain the difference between Lasso, Ridge, and Elastic Net regression.
7. When should we use Elastic Net instead of Lasso or Ridge?
8. What is the impact of the regularization parameter (1) in Logistic Regression?
9. What are the key assumptions of Logistic Regression?
10. What are some alternatives to Logistic Regression for classification tasks?
11. What are Classification Evaluation Metrics?
12. How does class imbalance affect Logistic Regression?
13. What is Hyperparameter Tuning in Logistic Regression?
14. What are different solvers in Logistic Regression? Which one should be used?
15. How is Logistic Regression extended for multiclass classification?
16. What are the advantages and disadvantages of Logistic Regression?
17. What are some use cases of Logistic Regression?
18. What is the difference between Softmax Regression and Logistic Regression?
19. How do we choose between One-vs-Rest (OvR) and Softmax for multiclass classification?
20. How do we interpret coefficients in Logistic Regression?

### Practical Questions

1. Write a Python program that loads a dataset, splits it into training and testing sets, applies Logistic Regression, and prints the model accuracy.
2. Write a Python program to apply Ll regularization (Lasso) on a dataset using LogisticRegression(penalty='1') and print the model accuracy,
3. Write a Python program to train Logistic Regression with L2 regularization (Ridge) using LogisticRegression(penalty=12). Print model accuracy and coefficients.
4. Write a Python program to train Logistic Regression with Elastic Net Regularization (penalty='elasticnet).
5. Write a Python program to train a Logistic Regression model for multiclass classification using multi_class=ovr
6. Write a Python program to apply GridSearchCV to tune the hyperparameters (C and penalty) of Logistic Regression. Print the best parameters and accuracy.
7. Write a Python program to evaluate Logistic Regression using Stratified K-Fold Cross-Validation. Print the average accuracy.
8. Write a Python program to load a dataset from a CSV file, apply Logistic Regression, and evaluate its accuracy.
9. Write a Python program to apply Randomized SearchCV for tuning hyperparameters (C, penalty, solver) in Logistic Regression. Print the best parameters and accuracy.
10. Write a Python program to implement One-vs-One (OvO) Multiclass Logistic Regression and print accuracy.
11. Write a Python program to train a Logistic Regression model and visualize the confusion matrix for binary classification.
12. Write a Python program to train a Logistic Regression model and evaluate its performance using Precision, Recall, and FI-Score.
13. Write a Python program to train a Logistic Regression model on imbalanced data and apply class weights to improve model performance.
14. Write a Python program to train Logistic Regression on the Titanic dataset, handle missing values, and evaluate performance.
15. Write a Python program to apply feature scaling (Standardization) before training a Logistic Regression model. Evaluate its accuracy and compare results with and without scaling.
16. Write a Python program to train Logistic Regression and evaluate its performance using ROC-AUC score.
17. Write a Python program to train Logistic Regression using a custom learning rate (C=0.5) and evaluate accuracy.
18. Write a Python program to train Logistic Regression and identify important features based on model coefficients.
19. Write a Python program to train Logistic Regression and evaluate its performance using Cohen's Kappa Score.
20. Write a Python program to train Logistic Regression and visualize the Precision-Recall Curve for binary classification
21. Write a Python program to train Logistic Regression with different solvers (liblinear, saga, Ibtgs) and compare their accuracy.
22. Write a Python program to train Logistic Regression and evaluate its performance using Matthews Correlation Coefficient (MCC).
23. Write a Python program to train Logistic Regression on both raw and standardized data. Compare their accuracy to see the impact of feature scaling
24. Write a Python program to train Logistic Regression and find the optimal C (regularization strength) using cross-validation.
25. Write a Python program to train Logistic Regression, save the trained model using joblib, and load it again to make predictions.
