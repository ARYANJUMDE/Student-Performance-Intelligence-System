import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

st.set_page_config(
    page_title="Student Performance Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data(path="data/student_data.csv"):
    return pd.read_csv(path)


@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    df["Result"] = (df["G3"] >= 10).astype(int)

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.astype(int)

    X = df_encoded.drop(["G3", "Result"], axis=1)
    y_class = df_encoded["Result"]
    y_reg = df_encoded["G3"]

    return df, df_encoded, X, y_class, y_reg, categorical_cols


@st.cache_resource
def train_classification_models(X, y_class):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(X_scaled, y_class)

    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt_grid = GridSearchCV(dt, {"max_depth": [3, 5, 7]}, cv=5)
    dt_grid.fit(X_scaled, y_class)
    best_dt = dt_grid.best_estimator_

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_grid = GridSearchCV(rf, {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]}, cv=5)
    rf_grid.fit(X_scaled, y_class)
    best_rf = rf_grid.best_estimator_

    return scaler, logistic, best_dt, best_rf, dt_grid.best_params_, rf_grid.best_params_


@st.cache_resource
def train_regression_models(X, y_reg):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    linear = LinearRegression()
    linear.fit(X_scaled, y_reg)

    dt = DecisionTreeRegressor(random_state=42)
    dt_grid = GridSearchCV(dt, {"max_depth": [3, 5, 7]}, cv=5)
    dt_grid.fit(X_scaled, y_reg)
    best_dt = dt_grid.best_estimator_

    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]}, cv=5)
    rf_grid.fit(X_scaled, y_reg)
    best_rf = rf_grid.best_estimator_

    return scaler, linear, best_dt, best_rf, dt_grid.best_params_, rf_grid.best_params_


@st.cache_resource
def train_clustering(df):
    cluster_features = ["studytime", "absences", "G1", "G2", "failures"]
    X_cluster = df[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    inertia = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    default_k = 6
    kmeans = KMeans(n_clusters=default_k, random_state=42)
    kmeans.fit(X_scaled)
    clusters = kmeans.labels_
    silhouette = silhouette_score(X_scaled, clusters)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return cluster_features, X_cluster, X_scaled, inertia, default_k, clusters, silhouette, pca, X_pca


def build_user_input(df, categorical_cols):
    st.sidebar.header("Student Input Parameters")

    numeric_features = {
        "age": (15, 22, 18),
        "traveltime": (1, 4, 1),
        "studytime": (1, 4, 2),
        "failures": (0, 3, 0),
        "famrel": (1, 5, 4),
        "freetime": (1, 5, 3),
        "goout": (1, 5, 3),
        "Dalc": (1, 5, 1),
        "Walc": (1, 5, 1),
        "health": (1, 5, 3),
        "absences": (0, 93, 4),
        "G1": (0, 20, 10),
        "G2": (0, 20, 10),
    }

    inputs = {}
    for feature, (low, high, default) in numeric_features.items():
        inputs[feature] = st.sidebar.slider(feature.title().replace("_", " "), low, high, default)

    for col in categorical_cols:
        options = sorted(df[col].dropna().unique())
        default = options[0] if len(options) else ""
        inputs[col] = st.sidebar.selectbox(col.title(), options, index=options.index(default))

    return inputs


def make_prediction(inputs, X_columns):
    data = pd.DataFrame([inputs])
    data_encoded = pd.get_dummies(data, drop_first=True)
    data_encoded = data_encoded.reindex(columns=X_columns, fill_value=0)
    return data_encoded


def display_data_overview(df):
    st.subheader("Dataset Overview")
    st.markdown(
        "A student performance dataset containing academic, demographic, and behavioral indicators. Explore key patterns, study time relationships, and grade correlations."
    )

    st.markdown("### Dataset sample")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Pass Rate", f"{(df['G3'] >= 10).mean() * 100:.1f}%")

    st.markdown("### Data diagnostics")
    st.write(df.describe())
    st.write("Missing values by column:")
    st.write(df.isnull().sum())

    st.markdown("---")
    st.subheader("Grade distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["G3"], bins=12, kde=True, color="#4c72b0", ax=ax)
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Study time vs final grade")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["studytime"], y=df["G3"], palette="Spectral", ax=ax)
    ax.set_xlabel("Study Time")
    ax.set_ylabel("Final Grade")
    st.pyplot(fig)

    st.subheader("Correlation matrix")
    numeric = df.select_dtypes(include=["int64", "float64"])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)


def evaluate_classification(clf, scaler, X, y_class):
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }

    prob = None
    roc_auc = None
    fpr = None
    tpr = None
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, prob)
        fpr, tpr, _ = roc_curve(y_test, prob)

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, roc_auc, fpr, tpr


def display_classification_results(X, y_class, scaler, logistic, dt, rf, dt_params, rf_params):
    st.subheader("Classification Analysis")
    st.markdown(
        "This section reproduces your classification notebook: logistic regression, decision tree, and random forest classification with hyperparameter tuning, cross-validation, feature importance, confusion matrices, and ROC curves."
    )

    X_scaled = scaler.transform(X)
    metrics_rows = []
    for name, model in [("Logistic Regression", logistic), ("Decision Tree", dt), ("Random Forest", rf)]:
        cv_scores = cross_val_score(model, X_scaled, y_class, cv=5, scoring="accuracy")
        metrics, _, roc_auc, _, _ = evaluate_classification(model, scaler, X, y_class)
        metrics_rows.append({
            "Model": name,
            **metrics,
            "ROC AUC": f"{roc_auc:.3f}" if roc_auc is not None else "N/A",
            "CV Mean": f"{cv_scores.mean():.3f}",
            "CV Std": f"{cv_scores.std():.3f}",
        })

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    st.write(metrics_df)
    best_f1_model = metrics_df["F1 Score"].astype(float).idxmax()
    st.markdown(f"**Best model by F1 score:** {best_f1_model}")
    st.markdown("#### Best hyperparameters")
    st.write({"Decision Tree": dt_params, "Random Forest": rf_params})

    model_choice = st.selectbox("Select model for deeper evaluation", ["Logistic Regression", "Decision Tree", "Random Forest"])
    model_map = {"Logistic Regression": logistic, "Decision Tree": dt, "Random Forest": rf}
    selected_model = model_map[model_choice]
    metrics, cm, roc_auc, fpr, tpr = evaluate_classification(selected_model, scaler, X, y_class)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Selected model metrics")
        for label, value in metrics.items():
            st.write(f"**{label}:** {value:.3f}")
        if roc_auc is not None:
            st.write(f"**ROC AUC:** {roc_auc:.3f}")
    with col2:
        st.markdown("##### Confusion matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    if fpr is not None and tpr is not None:
        st.markdown("##### ROC curve")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_choice}")
        ax.legend()
        st.pyplot(fig)

    if model_choice == "Random Forest":
        st.markdown("##### Feature importance")
        importances = pd.Series(selected_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:12]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
        ax.set_title("Top 12 Features for Random Forest")
        st.pyplot(fig)


def evaluate_regression(model, scaler, X, y_reg):
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return {
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }, y_test, y_pred


def display_regression_results(X, y_reg, scaler, linear, dt, rf, dt_params, rf_params):
    st.subheader("Regression Analysis")
    st.markdown(
        "This section reproduces your regression notebook: linear regression, decision tree, and random forest models with hyperparameter tuning, metrics, cross-validation, and actual vs predicted plots."
    )

    X_scaled = scaler.transform(X)
    metrics_rows = []
    for name, model in [("Linear Regression", linear), ("Decision Tree", dt), ("Random Forest", rf)]:
        cv_scores = cross_val_score(model, X_scaled, y_reg, cv=5, scoring="r2")
        metrics, _, _ = evaluate_regression(model, scaler, X, y_reg)
        metrics_rows.append({
            "Model": name,
            **metrics,
            "CV Mean R2": f"{cv_scores.mean():.3f}",
            "CV Std": f"{cv_scores.std():.3f}",
        })

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    st.write(metrics_df)
    best_r2_model = metrics_df["R2"].astype(float).idxmax()
    st.markdown(f"**Best model by R2 score:** {best_r2_model}")
    st.markdown("#### Best hyperparameters")
    st.write({"Decision Tree": dt_params, "Random Forest": rf_params})

    model_choice = st.selectbox("Select regression model for deeper evaluation", ["Linear Regression", "Decision Tree", "Random Forest"])
    model_map = {"Linear Regression": linear, "Decision Tree": dt, "Random Forest": rf}
    selected_model = model_map[model_choice]
    metrics, y_test, y_pred = evaluate_regression(selected_model, scaler, X, y_reg)

    st.markdown("##### Selected model metrics")
    st.write(metrics)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual G3")
    ax.set_ylabel("Predicted G3")
    ax.set_title(f"Actual vs Predicted - {model_choice}")
    st.pyplot(fig)

    if model_choice in ["Decision Tree", "Random Forest"]:
        st.markdown("##### Feature importance")
        importances = pd.Series(selected_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:12]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances.values, y=importances.index, palette="magma", ax=ax)
        ax.set_title(f"Top 12 Features - {model_choice}")
        st.pyplot(fig)


def display_clustering_results(df, cluster_features, X_scaled, inertia):
    st.subheader("Clustering Analysis")
    st.markdown(
        "This section reproduces your clustering notebook: the elbow method, silhouette score, k-means clusters, PCA projection, and cluster summaries."
    )

    st.markdown("#### Elbow Method")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 11), inertia, marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    k = st.slider("Choose number of clusters", 2, 10, 6)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    st.markdown(f"**Silhouette score for k={k}:** {score:.3f}")

    df_clusters = df[cluster_features].copy()
    df_clusters["Cluster"] = labels

    st.markdown("#### Cluster scatter: G1 vs G2")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df_clusters["G1"], y=df_clusters["G2"], hue=df_clusters["Cluster"], palette="deep", ax=ax)
    ax.set_title("Student Clusters")
    st.pyplot(fig)

    st.markdown("#### PCA projection")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="deep", ax=ax)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Cluster Visualization")
    st.pyplot(fig)
    st.write("Explained variance ratio:", pca.explained_variance_ratio_)

    st.markdown("#### Cluster counts and summaries")
    st.write(df_clusters["Cluster"].value_counts().sort_index())
    st.write(df_clusters.groupby("Cluster").median())


def main():
    st.title("📚 Student Performance Intelligence")
    st.markdown(
        "This app is built from your three notebooks and includes dataset exploration, classification results, regression results, clustering analysis, and student prediction."
    )

    df = load_data()
    df, df_encoded, X, y_class, y_reg, categorical_cols = preprocess_data(df)

    with st.spinner("Training models and preparing analysis..."):
        clf_scaler, logistic, dt_clf, rf_clf, dt_clf_params, rf_clf_params = train_classification_models(X, y_class)
        reg_scaler, linear, dt_reg, rf_reg, dt_reg_params, rf_reg_params = train_regression_models(X, y_reg)
        cluster_features, X_cluster, X_scaled, inertia, default_k, clusters, silhouette, pca, X_pca = train_clustering(df)

    page = st.sidebar.selectbox(
        "Choose analysis page",
        ["Overview", "Classification", "Regression", "Clustering", "Prediction"],
    )

    if page == "Overview":
        display_data_overview(df)
    elif page == "Classification":
        display_classification_results(X, y_class, clf_scaler, logistic, dt_clf, rf_clf, dt_clf_params, rf_clf_params)
    elif page == "Regression":
        display_regression_results(X, y_reg, reg_scaler, linear, dt_reg, rf_reg, dt_reg_params, rf_reg_params)
    elif page == "Clustering":
        display_clustering_results(df, cluster_features, X_scaled, inertia)
    else:
        st.subheader("Predict Student Outcomes")
        st.markdown("Enter a student profile and predict pass/fail status or the final grade.")
        inputs = build_user_input(df, categorical_cols)
        X_new = make_prediction(inputs, X.columns)

        predict_type = st.selectbox("Prediction type", ["Pass/Fail", "Final Grade"])
        if st.button("Predict"):
            if predict_type == "Pass/Fail":
                prob = rf_clf.predict_proba(clf_scaler.transform(X_new))[0][1]
                pred = rf_clf.predict(clf_scaler.transform(X_new))[0]
                st.metric("Prediction", "Pass" if pred == 1 else "Fail")
                st.metric("Pass probability", f"{prob * 100:.1f}%")
            else:
                grade = rf_reg.predict(reg_scaler.transform(X_new))[0]
                st.metric("Predicted final grade", f"{grade:.1f}/20")

            st.markdown("---")
            st.write("### Input summary")
            st.json(inputs)
        else:
            st.info("Press Predict after entering values.")

    st.markdown("---")
    st.caption("Built with Streamlit, pandas, scikit-learn, seaborn.")


if __name__ == "__main__":
    main()
