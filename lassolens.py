import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Removed duplicate import of LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


#Define a function to display evaluation metrics
def regression_results(y_true, y_pred, regr_type):

    # Regression metrics
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred) 
    mse = mean_squared_error(y_true, y_pred) 
    r2 = r2_score(y_true, y_pred)
    
    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('Explained Variance: ',  round(ev, 4)) 
    print('R²: ', round(r2, 4))
    print('MAE: ', round(mae, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print()



#Generate a simple dataset with one feature

# Generate synthetic data
noise=1
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + noise*np.random.randn(1000, 1)  # Linear relationship with some noise
y_ideal =  4 + 3 * X
# Specify the portion of the dataset to add outliers (e.g., the last 20%)
y_outlier = pd.Series(y.reshape(-1).copy())

# Identify indices where the feature variable X is greater than a certain threshold
threshold = 1.5  # Example threshold to add outliers for larger feature values
outlier_indices = np.where(X.flatten() > threshold)[0]

# Add outliers at random locations within the specified portion
num_outliers = 5  # Number of outliers to add
selected_indices = np.random.choice(outlier_indices, num_outliers, replace=False)

# Modify the target values at these indices to create outliers (add significant noise)
y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)

#Plot the data with outliers and the ideal fit line
plt.figure(figsize=(12, 6))

# Scatter plot of the data with outliers
plt.scatter(X, y_outlier, alpha=0.5, label='Data with Outliers')

# Plot the ideal, noise-free line
plt.plot(X, y_ideal, linewidth=2.5, color='g', label='Ideal (Noise-Free) Line')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Data with Outliers vs. Ideal Linear Relationship')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Task  1. Plot the data without the outliers and the ideal fit line
plt.figure(figsize=(12, 6))

# Scatter plot of the data without outliers
plt.scatter(X, y, alpha=0.4, edgecolor='k', label='Original Data (Without Outliers)')

# Plot the ideal, noise-free line
plt.plot(X, y_ideal, linewidth=3.5, color='g', label='Ideal (Noise-Free) Line')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Data Without Outliers vs. Ideal Linear Relationship')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Fit Ordinary, Ridge, and Lasso regression models and use them to make predicitions on the original, outlier-free data
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# Fit a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y_outlier)
y_outlier_pred_lin = lin_reg.predict(X)

# Fit a ridge regression model (regularization to control large coefficients)
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y_outlier)
y_outlier_pred_ridge = ridge_reg.predict(X)

# Fit a lasso regression model (regularization to control large coefficients)
lasso_reg = Lasso(alpha=.2)
lasso_reg.fit(X, y_outlier)
y_outlier_pred_lasso = lasso_reg.predict(X)

#Print the regression results


# The evaluation function is already defined earlier, so this duplicate definition is removed.

# Now run the evaluation
regression_results(y, y_outlier_pred_lin, 'Ordinary')
regression_results(y, y_outlier_pred_ridge, 'Ridge')
regression_results(y, y_outlier_pred_lasso, 'Lasso')

#Plot the data and the predictions for comparison
# Sort values for clean line plotting
sorted_indices = X.flatten().argsort()
X_sorted = X[sorted_indices]
y_ideal_sorted = y_ideal[sorted_indices]
y_outlier_pred_lin_sorted = y_outlier_pred_lin[sorted_indices]
y_outlier_pred_ridge_sorted = y_outlier_pred_ridge[sorted_indices]
y_outlier_pred_lasso_sorted = y_outlier_pred_lasso[sorted_indices]

plt.figure(figsize=(12, 6))

# Scatter plot of the original data (without outliers in the y values)
plt.scatter(X, y, alpha=0.4, edgecolor='k', label='Original Data')

# Ideal regression line
plt.plot(X_sorted, y_ideal_sorted, linewidth=2, color='k', label='Ideal, Noise-free Data')

# Linear regression prediction
plt.plot(X_sorted, y_outlier_pred_lin_sorted, linewidth=5, label='Linear Regression')

# Ridge regression prediction
plt.plot(X_sorted, y_outlier_pred_ridge_sorted, linestyle='--', linewidth=2, label='Ridge Regression')

# Lasso regression prediction
plt.plot(X_sorted, y_outlier_pred_lasso_sorted, linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with Outliers')
plt.legend()
plt.show()

#Task  2. Build the models and the prediction plots from the same data, excluding the outliers

# Fit a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Fit a ridge regression model
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y)
y_pred_ridge = ridge_reg.predict(X)

# Fit a lasso regression model
lasso_reg = Lasso(alpha=0.2)
lasso_reg.fit(X, y)
y_pred_lasso = lasso_reg.predict(X)

# Print the regression results
regression_results(y, y_pred_lin, 'Ordinary')
regression_results(y, y_pred_ridge, 'Ridge')
regression_results(y, y_pred_lasso, 'Lasso')

# Sort values for clean plotting
sorted_idx = X.flatten().argsort()
X_sorted = X[sorted_idx]
y_ideal_sorted = y_ideal[sorted_idx]
y_pred_lin_sorted = y_pred_lin[sorted_idx]
y_pred_ridge_sorted = y_pred_ridge[sorted_idx]
y_pred_lasso_sorted = y_pred_lasso[sorted_idx]

# Plot the data and the predictions
plt.figure(figsize=(12, 8))

# Scatter plot of the original data
plt.scatter(X, y, alpha=0.4, ec='k', label='Original Data')

# Ideal regression line (noise-free)
plt.plot(X_sorted, y_ideal_sorted, linewidth=2, color='k', label='Ideal, noise-free data')

# Linear Regression predictions
plt.plot(X_sorted, y_pred_lin_sorted, linewidth=5, label='Linear Regression')

# Ridge Regression predictions
plt.plot(X_sorted, y_pred_ridge_sorted, linestyle='--', linewidth=2, label='Ridge Regression')

# Lasso Regression predictions
plt.plot(X_sorted, y_pred_lasso_sorted, linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with No Outliers')
plt.legend()
plt.show()


#Multiple regression regularization and Lasso feature selction

# Create a high-dimensional regression dataset
X, y, ideal_coef = make_regression(
    n_samples=100, 
    n_features=100, 
    n_informative=10, 
    noise=10, 
    random_state=42, 
    coef=True
)

# Ideal predictions using the true underlying coefficients
ideal_predictions = X @ ideal_coef

# Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(
    X, y, ideal_predictions, test_size=0.3, random_state=42
)

#Initialize and fit the linear regression models and use them to predict the target
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

#Print the regression results
regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')

#TASK  3. Do you have some immediate thoughts on these performance metrics?
# Plotting the coefficients of all models
plt.figure(figsize=(16, 6))

# Linear Regression Coefficients
plt.subplot(1, 3, 1)
plt.stem(linear.coef_)
plt.title('Linear Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

# Ridge Regression Coefficients
plt.subplot(1, 3, 2)
plt.stem(ridge.coef_)
plt.title('Ridge Regression Coefficients')
plt.xlabel('Feature Index')

# Lasso Regression Coefficients
plt.subplot(1, 3, 3)
plt.stem(lasso.coef_)
plt.title('Lasso Regression Coefficients (Sparse)')
plt.xlabel('Feature Index')

plt.tight_layout()
plt.show()
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Lasso selected {len(selected_features)} features: {selected_features.tolist()}")

#Plot the predictions vs actuals

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0,0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,0].set_title("Linear Regression")
axes[0,0].set_xlabel("Actual",)
axes[0,0].set_ylabel("Predicted",)

# Removed redundant scatter plot for Lasso regression to avoid duplication

axes[0,1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,1].set_title("Ridge Regression",)
axes[0,1].set_xlabel("Actual",)

axes[0,2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0,2].set_title("Lasso Regression",)
axes[0,2].set_xlabel("Actual",)


# Line plots for predictions compared to actual and ideal predictions
axes[1,0].plot(y_test, label="Actual", lw=2)
axes[1,0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1,0].set_title("Linear vs Ideal",)
axes[1,0].legend()
 
axes[1,1].plot(y_test, label="Actual", lw=2)
# axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
axes[1,1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1,1].set_title("Ridge vs Ideal",)
axes[1,1].legend()
 
axes[1,2].plot(y_test, label="Actual", lw=2)
axes[1,2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1,2].set_title("Lasso vs Ideal",)
axes[1,2].legend()
 
plt.tight_layout()
plt.show()

#Model coefficients

# Model coefficients
linear_coeff = linear.coef_
ridge_coeff = ridge.coef_
lasso_coeff = lasso.coef_


# Plot the coefficients
x_axis = np.arange(len(linear_coeff))
x_labels = np.arange(min(x_axis),max(x_axis),10)
plt.figure(figsize=(12, 6))

plt.scatter(x_axis, ideal_coef,  label='Ideal', color='blue', ec='k', alpha=0.4)
plt.bar(x_axis - 0.25, linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ridge_coeff, width=0.25, label='Ridge Regression', color='green')
plt.bar(x_axis + 0.25, lasso_coeff, width=0.25, label='Lasso Regression', color='red')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Model Coefficients')
plt.xticks(x_labels)
plt.legend()
plt.show()


# Plot the coefficient residuals
x_axis = np.arange(len(linear_coeff))

plt.figure(figsize=(12, 6))

plt.bar(x_axis - 0.25, ideal_coef - linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ideal_coef - ridge_coeff, width=0.25, label='Ridge Regression', color='green')
# plt.bar(x_axis + 0.25, ideal_coef - lasso_coeff, width=0.25, label='Lasso Regression', color='red')
plt.plot(x_axis, ideal_coef - lasso_coeff, label='Lasso Regression', color='red')

plt.bar(x_axis + 0.25, ideal_coef - lasso_coeff, width=0.25, label='Lasso Regression', color='red')
# plt.plot(x_axis, ideal_coef - lasso_coeff, label='Lasso Regression', color='red')
plt.title('Comparison of Model Coefficient Residuals')
plt.xticks(x_labels)
plt.legend()
plt.show()

#Use Lasso to select the most important features and compare the three different linear regression models again on the resulting data
# Set threshold based on visual inspection (or trial and error)
# Threshold value chosen based on visual inspection or trial and error to identify significant features
threshold = 5

# Create DataFrame comparing Lasso and Ideal Coefficients
feature_importance_df = pd.DataFrame({
    'Lasso Coefficient': lasso_coeff,
    'Ideal Coefficient': ideal_coef
})

# Mark features as selected if Lasso coefficient magnitude is greater than threshold
FEATURE_SELECTED = 'Feature Selected'
feature_importance_df[FEATURE_SELECTED] = feature_importance_df['Lasso Coefficient'].abs() > threshold

# Display important features identified by Lasso
print("Important features identified by Lasso:")
print(feature_importance_df[feature_importance_df[FEATURE_SELECTED]])
print(feature_importance_df[feature_importance_df[FEATURE_SELECTED]])

# Display all features with nonzero ideal coefficients
print("\n✅ Nonzero Ideal Coefficient Indices (Ground Truth):")
print(feature_importance_df[feature_importance_df['Ideal Coefficient'] != 0])

important_features = feature_importance_df[feature_importance_df[FEATURE_SELECTED]].index
important_features = feature_importance_df[feature_importance_df['Feature Selected']].index


# Filter features
X_filtered = X[:, important_features]
print("Shape of the filtered feature set:", X_filtered.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X_filtered, y, ideal_predictions, test_size=0.3, random_state=42)

#Part 3. Fit and apply the three models to the selected features
# Removed redundant evaluate_model function and its calls.

#Task  4. Print the regression performance results
# Print the regression performance results
regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')

#Task  5. Regenerate the same plots as before and compare the results

# Plot the predictions vs actuals
# Plot predictions vs actuals for models trained on selected features
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

# --- SCATTER PLOTS ---

axes[0, 0].scatter(y_test, y_pred_linear, color="red")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 0].set_title("Linear Regression (Filtered)")
axes[0, 0].set_xlabel("Actual")
axes[0, 0].set_ylabel("Predicted")

axes[0, 1].scatter(y_test, y_pred_ridge, color="green")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 1].set_title("Ridge Regression (Filtered)")
axes[0, 0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Ideal Fit")
axes[0, 0].set_title("Linear Regression (Filtered)")
axes[0, 0].set_xlabel("Actual Values")
axes[0, 0].set_ylabel("Predicted Values")
axes[0, 0].legend()

axes[0, 1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Ideal Fit")
axes[0, 1].set_title("Ridge Regression (Filtered)")
axes[0, 1].set_xlabel("Actual Values")
axes[0, 1].legend()

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Ideal Fit")
axes[0, 2].set_title("Lasso Regression (Filtered)")
axes[0, 2].set_xlabel("Actual Values")
axes[0, 2].legend()
axes[1, 1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1, 1].set_title("Ridge vs Actual")
axes[1, 1].legend()

axes[1, 2].plot(y_test, label="Actual", lw=2)
axes[1, 2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1, 2].set_title("Lasso vs Actual")
axes[1, 2].legend()

plt.tight_layout()
plt.show()
