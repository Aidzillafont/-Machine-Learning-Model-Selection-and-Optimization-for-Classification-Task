# Machine Learning Model Selection and Optimization for Classification Task

This project aims to build an optimal machine learning model for a classification task by exploring and comparing various classifiers and applying techniques to enhance their performance. The project covers feature selection, sparsity evaluation, pipeline creation, and parameter tuning through cross-validation.

## Classifiers Explored

1. K-Nearest Neighbors (KNN)
2. Decision Tree
3. Random Forest
4. Extreme Gradient Boosting (XGBoost)

## Pipeline

The pipeline includes:

1. Custom transformation `RemoveInf()` to replace infinite values with NaNs.
2. Imputation using `SimpleImputer()` to replace NaNs with the mean value of the corresponding feature.
3. Feature scaling using `StandardScaler()`.
4. Dimensionality reduction using `PCA`.
5. Estimator (classifier) of choice.

## Parameter Selection and Cross-Validation

Parameters for each classifier are tuned using GridSearchCV (or RandomizedSearchCV for faster results). The classifiers' hyperparameters are selected based on their performance during cross-validation.

## Results

Among the classifiers explored, XGBoost demonstrated the best performance. As the test and training scores are closely aligned and XGBoost incorporates techniques to reduce overfitting, this model is chosen as the most suitable for the given task.

A copy of the best estimator has been uploaded to GitHub for the grading sections. Please note that the final cell imports additional libraries along with the custom transformer, allowing it to be executed after restarting the kernel.

## How to Run

1. Clone the repository.
2. Open the Jupyter Notebook in your preferred environment.
3. Run all cells in the notebook, making sure to restart the kernel when prompted in the instructions.
