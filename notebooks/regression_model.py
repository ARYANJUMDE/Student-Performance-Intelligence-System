import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the dataset

df=pd.read_csv("data\student_data.csv")

# Data Preprocessing

# Information about dataset

print(df.info())

# Mathematical analysis of dataset

print(df.describe())

# Top 5 rows of dataset

print(df.head())

# Columns in dataset

print(df.columns)

# Checking for missing values

print(df.isnull().sum())

# Removing rows with missing values

df.dropna(inplace=True)

# Converting categorical variables to numerical using one-hot encoding

df_categorical=df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=df_categorical, drop_first=True)
df = df.astype(int)

# Define features and target variable

X=df.drop("G3",axis=1)
Y=df["G3"]

# Train test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Model's Training and Evaluation

# Linear Regression

linear_model=LinearRegression()
linear_model.fit(X_train_scaled,Y_train)
Y_pred_linear=linear_model.predict(X_test_scaled)

# Random Forest Regressor

RandomForest_model=RandomForestRegressor(random_state=42)

# Hyperparameter Tuning

param_grid={
    "n_estimators":[50,100,200],
    "max_depth":[3,5,7]
}

grid_search_model_RandomForest=GridSearchCV(estimator=RandomForest_model,param_grid=param_grid,cv=5)

grid_search_model_RandomForest.fit(X_train,Y_train)

# Best Hyperparameters

print("Selected Hyperparametrs:", grid_search_model_RandomForest.best_params_)

# Printing Score 

print("Best Score for Cross Validation:",grid_search_model_RandomForest.best_score_)

# Best Model
best_model_RandomForest=grid_search_model_RandomForest.best_estimator_

Y_pred_RandomForest=best_model_RandomForest.predict(X_test)

# Decision Tree Regressor

DecisionTree_model=DecisionTreeRegressor(random_state=42)

param_grid={
    "max_depth":[3,5,7]
}
grid_search_model_Decision_Tree=GridSearchCV(estimator=DecisionTree_model,param_grid=param_grid,cv=5)

grid_search_model_Decision_Tree.fit(X_train,Y_train)

# Best Hyperparameters

print("Selected Hyperparametrs:", grid_search_model_Decision_Tree.best_params_)

# Best Score

print("Selected Hyperparametrs:", grid_search_model_Decision_Tree.best_score_)

best_model_DecisionTree=grid_search_model_Decision_Tree.best_estimator_

Y_pred_DecisionTree=best_model_DecisionTree.predict(X_test)

# Evaluation Metrics

# Linear Regression

mean_squared_error_linear_model=mean_squared_error(Y_test,Y_pred_linear)
r2_score_linear_model=r2_score(Y_test,Y_pred_linear)

# RandomForest Regression

mean_squared_error_RandomForest_model=mean_squared_error(Y_test,Y_pred_RandomForest)
r2_score_RandomForest_model=r2_score(Y_test,Y_pred_RandomForest)

# DecisionTree Regression

mean_squared_error_DecisionTree_model=mean_squared_error(Y_test,Y_pred_DecisionTree)
r2_score_DecisionTree_model=r2_score(Y_test,Y_pred_DecisionTree)

# Comparing Model Metrics

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Mean Squared Error": [
        mean_squared_error_linear_model,
        mean_squared_error_RandomForest_model,
        mean_squared_error_DecisionTree_model
    ],
    "R2 Score": [
        r2_score_linear_model,
        r2_score_RandomForest_model,
        r2_score_DecisionTree_model
    ]
})

print(results)


# Graphs

plt.figure(figsize=(10,6))
sns.barplot(x="Model",y="R2 Score",data=results)
plt.title("Model Comparision Based of R2 Score")
plt.show()

# Best Model

best_model=results.loc[results["R2 Score"].idxmax()]
print("Best Model:",best_model)

# Cross Validation Score

# Linear Regression

cross_val_score_Linear_Regression=cross_val_score(linear_model,X_train_scaled,Y_train,cv=5)
print("cross_val_score for Linear Regression:",cross_val_score_Linear_Regression)
print("Mean of cross_val_score for Linear Regression:",cross_val_score_Linear_Regression.mean())
print("Standard Deviation of cross_val_score for Linear Regression:",cross_val_score_Linear_Regression.std())

# RandomForest Regressor

cross_val_score_RandomForest_Regressor=cross_val_score(best_model_RandomForest,X_train,Y_train,cv=5)
print("cross_val_score for RandomForest Regressor:",cross_val_score_RandomForest_Regressor)
print("Mean of cross_val_score for RandomForest Regressor:",cross_val_score_RandomForest_Regressor.mean())
print("Standard Deviation of cross_val_score for RandomForest Regressor",cross_val_score_RandomForest_Regressor.std())

# DecisionTree Regressor

cross_val_score_DecisionTree_Regressor=cross_val_score(best_model_DecisionTree,X_train,Y_train)
print("cross_val_score for DecisionTree Regressor:",cross_val_score_DecisionTree_Regressor)
print("Mean of cross_val_score for DecisionTree Regressor:",cross_val_score_DecisionTree_Regressor.mean())
print("Standard Deviation of cross_val_score for DecisionTree Regressor:",cross_val_score_DecisionTree_Regressor.std())

# Actual Vs Predicted 

# Linear Regression

plt.figure(figsize=(10,6))
sns.scatterplot(x=Y_test,y=Y_pred_linear)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Vs Predicted Linear Regression")
plt.show()

# RandomForest Regressor

plt.figure(figsize=(10,6))
sns.scatterplot(x=Y_test,y=Y_pred_RandomForest)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Vs Predicted Random Forest")
plt.show()

# DecisionTree Regressor

plt.figure(figsize=(10,6))
sns.scatterplot(x=Y_test,y=Y_pred_DecisionTree)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Vs Predicted Decision Tree")
plt.show()


# Importance of Features

# Random Forest Regressor

print("Feature Importance for Random Forest")
importance = best_model_RandomForest.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance.head(10))

# DecisionTree Regressor

print("Feature Importance for Decision Tree")
importance = best_model_DecisionTree.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance.head(10))