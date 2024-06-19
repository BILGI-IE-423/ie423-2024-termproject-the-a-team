# Merging dataframes except merged_region_expectancy
Five dataframes were merged except one because we couldn’t merge the region one so we kept it separate since this goes by region names and not country names.
The plots show that there are outliers with skewness of data as well as the correlation matrix indicates that there are no strong linear relationships between any pairs of variables in the dataset. The relationships that do exist are very weak, suggesting that the variables are largely independent of each other. The strongest correlation is -0.19 between Period and Indicator_encoded, but this is still considered weak.
Thus, outliers that fall out of the interquartile range were removed.

# Split & Scale Merged Data
1- splitting data: the data is divided into training 80% and testing 20% sets. X_train and y_train are used to train the model, while X_test and y_test are used to evaluate its performance.
Purpose: To divide the dataset into training and testing sets for model evaluation.
2- scaling data: The feature values in X_train and X_test are standardized, ensuring that all features contribute equally to the model's performance. 
purpose: To standardize the features to have a mean of 0 and a standard deviation of 1, which helps many machine learning algorithms perform better. 

# Cross Validation
Several Models were applied in the cross-validation process to evaluate the performances of models. The incentive is to find the best performing model that gives the lowest MSE and then train and test on that model. A baseline model was used as a benchmark in cross validation, training, and testing. The Merged Dataframe without Region consists of a large dataset so dimensionality reduction was applied using PCA on linear regression and SVM(RBF kernel) and compared to regular linear regression and SVM(RBF kernel) without DR.
The Region dataframe consists of one csv file with preprocessed data, thus explaining why the values are smaller than the other data frame.

## Cross Validation on the Merged Dataframe

We analyzed the performance of multiple machine learning models on the merged dataset, focusing on Linear Regression, Support Vector Machine (SVM) with different kernels, Random Forest, Gradient Boosting, and a baseline DummyRegressor. The data was initially divided into training and test sets, and scaling was used to normalize the training data to zero mean and unit variance. This preprocessing step assured good performance for models that are sensitive to feature scales, such as SVMs. The baseline model was used as a ben

The Linear Regression model was trained and analyzed using 5-fold cross-validation, which involved dividing the training data into five subsets, training the model on four subsets, and validating it on the remaining one. This process was repeated five times to ensure robust performance estimation, resulting in a mean cross-validation mean squared error (MSE) of 1584.02.

For the SVM models, three different kernels were explored: linear, polynomial, and radial basis function (RBF). The SVM with a linear kernel yielded a mean CV MSE of 1667.13, while the polynomial kernel SVM (degre 3) resulted in a mean CV MSE of 1612.66. The RBF kernel SVM involved a grid search over C and gamma parameters, identifying C=1 and gamma=1 as the best parameters with a mean CV MSE of 1519.39.

Principal Component Analysis (PCA) was applied to reduce the data's dimensionality, retaining 95% of the variance. Models were retrained on the PCA-transformed data, but the performance did not significantly improve, as evidenced by the nearly identical mean CV MSE for Linear Regression with and without PCA.

The Random Forest regressor was optimized using a grid search over parameters such as the number of estimators, maximum depth, and minimum samples required for splitting and leaf nodes. The best parameters (max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100) resulted in a mean CV MSE of 460.82, demonstrating substantial predictive power. Similarly, the Gradient Boosting regressor was fine-tuned with a grid search, achieving a mean CV MSE of 480.52 with the best parameters (learning_rate=0.2, max_depth=5, min_samples_leaf=2, min_samples_split=2, n_estimators=200).

We can see that Random Forest and Gradient Boosting outperformed Linear Regression and SVMs, with Random Forest achieving the lowest mean CV MSE, indicating its robust predictive capabilities. Despite applying PCA, the original feature set sufficed for effective model training, underscoring the effectiveness of ensemble approaches in this context.

## Cross Validation for merged_region_expectancy

We further evaluated the performance of these machine learning models focusing on merged regional expectancy. The models included Random Forest, Gradient Boosting, and Support Vector Regressor (SVR), alongside the baseline DummyRegressor. Data preprocessing involved scaling the training data to standardize it, ensuring optimal performance for the models. 

A Random Forest regressor was configured with the optimal hyperparameters (max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=100) determined through prior tuning. Using 5-fold cross-validation, the model achieved a mean cross-validation mean squared error (MSE) of 21.70, indicating its accuracy.

An initial Gradient Boosting model with parameters (n_estimators=100, learning_rate=0.1, max_depth=3) was evaluated, yielding a mean CV MSE of 7.56. To refine this model, a grid search was conducted over a parameter grid, resulting in the best configuration (learning_rate=0.1, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=200). This optimized Gradient Boosting model significantly outperformed the initial model, achieving a mean CV MSE of 7.11.

An SVR with an RBF kernel was evaluated using optimal parameters (C=1, gamma=1, epsilon=0.01). The SVR's mean CV MSE was 27.90, indicating that while it performed reasonably well, it was less effective compared to the Random Forest and Gradient Boosting models.

A Dummy Regressor with the 'mean' strategy served as the baseline model. This model produced a mean CV MSE of 26.86, providing a reference point to assess the relative performance of the other models. This model 

Gradient Boosting model with optimized parameters demonstrated the best performance with the lowest mean CV MSE of 7.11, highlighting its superior predictive capabilities for the dataset. The Random Forest model also performed well, whereas the SVR and Dummy Regressor lagged behind.
Train
Based on these comparisons, the Random Forest and Gradient Boosting models are clearly outperforming the other models like Linear Regression and SVM(linear, polynomial, and RBF).
Also baseline model was used as a benchmark to compare cv mses with the mean predictor(1732.4501327715993 for merged dataframe and 26.86462607157178 for merged_region_expectancy) and proved that Random Forest and Gradient Boosting are better.
Both models are strong candidates for final model selection.
However, the Random Forest model has a slightly better CV MSE compared to Gradient Boosting in the MERGED DATAFRAME in which Best Gradient Boosting CV MSE: 480.5232280836393 and Best Random Forest CV MSE: 460.82440685038466.
Random Forest Mean CV MSE with merged_region_expectancy: 21.70019408286778 and Gradient Boosting Mean CV MSE with merged_region_expectancy: 7.560402962473326.
Gradient Boosting will be chosen for both datasets because: 
1- It has a significantly lower CV MSE for the merged region expectancy data. 
2- Its performance on the merged dataframe is very close to that of Random Forest, with only a small difference in CV MSE.
3- Gradient Boosting is often more robust and provides better generalization performance
So initialised Gradient Boosting with the best hyperparameters and then fit on the train data and then predict to calculate test. 

## Analysis on Model Performance: Train & Test Results
1. Merged Dataframe without Region
Mean Squared Error (MSE):
The MSE of 473.68957279280164 indicates that the model has moderate error in its predictions. While this is not exceedingly high, there is room for improvement.
Mean Absolute Error (MAE):
The MAE of 15.618870881691729 suggests that, on average, the model's predictions are off by approximately 15.6 units, which indicates moderate prediction accuracy.
R²:
An R² value of 0.7222439811184465 means that the model explains about 72.2% of the variance in the target variable. This indicates a good fit, but there is still 27.8% of the variance that is unexplained.
2. Merged Region Dataframe
Mean Squared Error (MSE):
The MSE of 2.714763837982076 is very low, suggesting that the model has high accuracy in its predictions.
Mean Absolute Error (MAE):
The MAE of 1.2427639145181417 indicates that, on average, the model's predictions are very close to the actual values.
R²:
An R² value of 0.46612022553671184 means that the model explains approximately 46.6% of the variance in the target variable. While this is moderate, it suggests that there is significant unexplained variance. That is probably due to the small dataframe.

## Analysis Based on Figures

1. Merged Dataframe without Region
Residuals Plot:
Observation: The residuals are scattered around the zero line, indicating that the model has captured the underlying pattern fairly well. However, there are some outliers and a funnel shape, suggesting heteroscedasticity (variance of errors is not constant). This is due to the large data, thus applying polynomial feature expansion later might address this. Applying log transformation did not deal with the outliers and skewness so opted against. 
Implication: The model's predictions are generally accurate but may struggle with certain ranges of the target variable, especially with higher values where residuals are larger.
Actual vs Predicted Plot:
Observation: The points are closely clustered around the diagonal line, showing a strong correlation between actual and predicted values.
Implication: The model predicts the target variable with good accuracy. The spread around the line indicates minor deviations between actual and predicted values.

2. Merged Region Dataframe
Residuals Plot:
Observation: The residuals show more scatter and less concentration around the zero line compared to the merged dataframe without the region. There are noticeable outliers.
Implication: The model's predictions have some error variance that is not constant across all values, indicating possible areas for improvement in capturing the target variable's behavior.
Actual vs Predicted Plot:
Observation: The points are somewhat dispersed around the diagonal line, indicating a moderate correlation between actual and predicted values. The dispersion suggests that while the model is reasonably accurate, there is room for improvement.
Implication: The model's predictive accuracy is moderate. The spread indicates that some predictions are further from the actual values, which might be improved by further tuning or incorporating additional features.

# Conclusion

## Conclusion For the Merged Dataframe without Region:
The Gradient Boosting model demonstrates a good fit with an R² value of 0.722 and reasonable error metrics. This model could be considered robust for predicting the target variable in this context.
For the Merged Region Dataframe:
The Gradient Boosting model has very low MSE and MAE, indicating high prediction accuracy. However, the R² value of 0.466 suggests that there is still substantial variance that the model does not explain. This could be improved with further feature engineering or by incorporating additional relevant features.
However, Gradient Boosting seems to be performing well in both dataframes give low MSE error, lower than in cross validation and baseline models. 

## Conclusion for Merged Dataframe without Region:
The Gradient Boosting model is performing well, with relatively high accuracy and good predictive performance. The residuals and actual vs predicted plots both suggest that the model captures the underlying pattern well, despite some heteroscedasticity.
Merged Region Dataframe:
The model shows reasonable accuracy but could benefit from further tuning. The residuals plot indicates some variance in errors, and the actual vs predicted plot suggests that the model's predictions could be more tightly clustered around the actual values.



![Screenshot 1](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/1.png)
![Screenshot 2](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/2.png)
![Screenshot 3](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/3.png)
![Screenshot 4](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/4.png)
![Screenshot 5](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/5.png)
![Screenshot 6](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/6.png)
![Screenshot 7](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/7.png)
![Screenshot 8](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/8.png)
![Screenshot 9](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/9.png)
![Screenshot 10](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/10.png)
![Screenshot 11](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/11.png)
![Screenshot 12](https://raw.githubusercontent.com/BILGI-IE-423/ie423-2024-termproject-the-a-team/6e8c0f93d22e8c39e1e51aae723845019855fe50/Preprocessing/Screenshots/12.png)
