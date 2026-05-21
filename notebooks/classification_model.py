import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
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

# Creating a new column "Result" which is 1 if G3 >= 10 else 0

df["Result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Selecting feature and target variable

X=df.drop(["G3","Result"],axis=1)
Y=df["Result"]

# Train test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)

# Scaling

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Model's Training and Evaluation

# Logistic Regression

logistic_model=LogisticRegression()
logistic_model.fit(X_train_scaled,Y_train)
Y_pred_logistic=logistic_model.predict(X_test_scaled)

# RandomForest Classifier

RandomForest_model=RandomForestClassifier(random_state=42,class_weight="balanced")

# HyperParameter Tuning

param_grid_Random_Forest={
    "n_estimators":[100,200,300],
    "max_depth":[3,5,7]
}

param_grid_model_RandomForest=GridSearchCV(estimator=RandomForest_model,param_grid=param_grid_Random_Forest,cv=5)

param_grid_model_RandomForest.fit(X_train,Y_train)

# Printing Best Hyperparameters

print("Selected HyperParameters:",param_grid_model_RandomForest.best_params_)

# Printing Score

print("Best Score for Cross Validation:",param_grid_model_RandomForest.best_score_)

# Best Model

best_model_RandomForest=param_grid_model_RandomForest.best_estimator_

Y_pred_RandomForest=best_model_RandomForest.predict(X_test)

# Decision Tree Classifier

DecisionTree_model=DecisionTreeClassifier(random_state=42,class_weight="balanced")

param_grid_Decsion_Tree={
    "max_depth":[3,5,7]
}

param_grid_model_Decision_Tree=GridSearchCV(estimator=DecisionTree_model,param_grid=param_grid_Decsion_Tree,cv=5)

param_grid_model_Decision_Tree.fit(X_train,Y_train)

# Printing Best Hyperparameters

print("Selected HyperParameters:",param_grid_model_Decision_Tree.best_params_)

# Printing Score

print("Best Score for Cross Validation:",param_grid_model_Decision_Tree.best_score_)

# Best Model

best_model_Decision_Tree=param_grid_model_Decision_Tree.best_estimator_

Y_pred_Decision_Tree=best_model_Decision_Tree.predict(X_test)

# Evaluation Metrics

# Logistic Regression

accuracy_score_logistic_model=accuracy_score(Y_test,Y_pred_logistic)
f1_score_logistic_model=f1_score(Y_test,Y_pred_logistic)
recall_score_logistic_model=recall_score(Y_test,Y_pred_logistic)
precision_score_logistic_model=precision_score(Y_test,Y_pred_logistic)

# RandomForest Regression

accuracy_score_RandomForest_model=accuracy_score(Y_test,Y_pred_RandomForest)
f1_score_RandomForest_model=f1_score(Y_test,Y_pred_RandomForest)
recall_score_RandomForest_model=recall_score(Y_test,Y_pred_RandomForest)
precision_score_RandomForest_model=precision_score(Y_test,Y_pred_RandomForest)

# DecisionTree Regression

accuracy_score_DecisionTree_model=accuracy_score(Y_test,Y_pred_Decision_Tree)
f1_score_DecisionTree_model=f1_score(Y_test,Y_pred_Decision_Tree)
recall_score_DecisionTree_model=recall_score(Y_test,Y_pred_Decision_Tree)
precision_score_DecisionTree_model=precision_score(Y_test,Y_pred_Decision_Tree)

# Comparing Model Metrics

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score_logistic_model,
        accuracy_score_RandomForest_model,
        accuracy_score_DecisionTree_model
    ],
    "F1 Score": [
        f1_score_logistic_model,
        f1_score_RandomForest_model,
        f1_score_DecisionTree_model
    ],
    "Recall Score": [
        recall_score_logistic_model,
        recall_score_RandomForest_model,
        recall_score_DecisionTree_model
    ],
    "Precision Score": [
        precision_score_logistic_model,
        precision_score_RandomForest_model,
        precision_score_DecisionTree_model
    ],
    
})

print(results)

# Graphs

plt.figure(figsize=(10,6))
sns.barplot(x="Model",y="F1 Score",data=results)
plt.title("Model Comparision Based of F1 Score")
plt.show()

# Best Model

best_model=results.loc[results["F1 Score"].idxmax()]
print("Best Model:",best_model)

# Cross Validation Score

# Logistic Regression

cross_val_score_Logistic_Regression=cross_val_score(logistic_model,X_train_scaled,Y_train,cv=5)
print("cross_val_score for Logistic Regression:",cross_val_score_Logistic_Regression)
print("Mean of cross_val_score for Logistic Regression:",cross_val_score_Logistic_Regression.mean())
print("Standard Deviation of cross_val_score for Logistic Regression:",cross_val_score_Logistic_Regression.std())

# RandomForest Classifier

cross_val_score_RandomForest_Classifier=cross_val_score(best_model_RandomForest,X_train,Y_train,cv=5)
print("cross_val_score for RandomForest Classifier:",cross_val_score_RandomForest_Classifier)
print("Mean of cross_val_score for RandomForest Classifier:",cross_val_score_RandomForest_Classifier.mean())
print("Standard Deviation of cross_val_score for RandomForest Classifier",cross_val_score_RandomForest_Classifier.std())

# DecisionTree Classifier

cross_val_score_DecisionTree_Classifier=cross_val_score(best_model_Decision_Tree,X_train,Y_train)
print("cross_val_score for DecisionTree Classifier:",cross_val_score_DecisionTree_Classifier)
print("Mean of cross_val_score for DecisionTree Classifier:",cross_val_score_DecisionTree_Classifier.mean())
print("Standard Deviation of cross_val_score for DecisionTree Classifier:",cross_val_score_DecisionTree_Classifier.std())

# Importance of Features

# Random Forest Classifier

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

# DecisionTree Classifier

print("Feature Importance for Decision Tree")
importance = best_model_Decision_Tree.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance.head(10))

# Confusion Matrix Logistic Regression

logistic_cm = confusion_matrix(
    Y_test,
    Y_pred_logistic
)
sns.heatmap(
    logistic_cm,
    annot=True,
    fmt="d",
    cmap="Reds"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# Confusion Matrix RandomForest

random_forest_cm = confusion_matrix(
    Y_test,
    Y_pred_RandomForest
)
print(random_forest_cm)
sns.heatmap(
    random_forest_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Confusion Matrix for Decision Tree Classifier

decision_tree_cm = confusion_matrix(
    Y_test,
    Y_pred_Decision_Tree
)
print(decision_tree_cm)
sns.heatmap(
    decision_tree_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ROC and AUC curve

# Logistic Regression

logistic_proba=logistic_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, logistic_proba)
auc = roc_auc_score(Y_test, logistic_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Logistic Regression")
plt.show()

# Random Forest

random_forest_proba=best_model_RandomForest.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, random_forest_proba)
auc = roc_auc_score(Y_test, random_forest_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Random Forest")
plt.show()

# Decision Tree

decision_tree_proba=best_model_Decision_Tree.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, decision_tree_proba)
auc = roc_auc_score(Y_test, decision_tree_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Decision Tree")
plt.show()