# 🎓 Student Performance Intelligence System

A complete Machine Learning project that analyzes student academic performance using Regression, Classification, and Clustering techniques, plus a Streamlit web app for interactive exploration.

This repository includes:
- Data preprocessing and analysis
- Regression models for grade prediction
- Classification models for pass/fail prediction
- K-Means clustering for student grouping
- Hyperparameter tuning with GridSearchCV
- Cross-validation and model comparison
- Interactive Streamlit deployment

---

# 🚀 What You Get

## 📈 Regression Analysis
Predicts students' final grades (`G3`) using:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Evaluation Metrics
- Mean Squared Error (MSE)
- R² Score
- Cross Validation Score

### Visualizations
- Actual vs Predicted scatter plot
- Feature importance bar chart

---

## 📚 Classification Analysis
Predicts whether a student will pass or fail.

### Models Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

### Visualizations
- Confusion matrix
- ROC curve
- Feature importance

---

## 🧠 Clustering Analysis
Groups students using features such as:
- Study time
- Absences
- Previous grades (G1, G2)
- Failures

### Techniques Used
- KMeans Clustering
- Elbow Method
- Silhouette Score
- PCA

### Visualizations
- Cluster scatter plot
- PCA cluster projection
- Cluster summary statistics

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

# 📂 Project Structure

```text
student-performance-intelligence-system/
├── app.py
├── data/
│   └── student_data.csv
├── notebooks/
│   ├── regression_model.py
│   ├── classification_model.py
│   └── clustering_model.py
├── Readme.md
```

> Note: The Streamlit app file `app.py` is located at the project root.

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone <your-repository-link>
```

## 2️⃣ Navigate to Project Folder

```bash
cd "student-performance-intelligence-system - Copy"
```

## 3️⃣ Install Dependencies

Install dependencies from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit App

From the repository root, run:

```bash
streamlit run app.py
```

---

# 📊 Machine Learning Concepts Implemented

## Supervised Learning
- Regression
- Classification

## Unsupervised Learning
- KMeans Clustering
- PCA

## Model Optimization
- Cross Validation
- GridSearchCV
- Hyperparameter Tuning

## Evaluation Techniques
- ROC-AUC
- Confusion Matrix
- Precision / Recall / F1
- Silhouette Score

---

# 📈 Results

This project compares multiple ML models and highlights the best models using evaluation metrics, graphs, and interactive analysis.

---

# 🎯 Future Improvements

- Deploy the app to Streamlit Cloud or another hosting platform
- Add model persistence with Joblib
- Add a full user input prediction dashboard
- Add dark mode support
- Add real-time student analytics

---

# 👨‍💻 Author

A portfolio ML project demonstrating:
- End-to-end data science workflow
- Model evaluation and comparison
- Visualization and interactive dashboards
- Deployment with Streamlit

---

# ⭐ Key Highlights

- End-to-End ML Project
- Regression + Classification + Clustering
- PCA Visualization
- Streamlit Deployment
- Hyperparameter Tuning
- Cross Validation
- Feature Importance Analysis
- Clear, interactive project structure