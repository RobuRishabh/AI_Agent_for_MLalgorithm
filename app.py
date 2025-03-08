import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from code_generator import generate_code
import os

# Streamlit UI Setup
st.set_page_config(page_title="AI ML Agent", layout="wide")
st.title("🤖 Code Generation ML Agent with Ollama ")

# Function to check if target is suitable for task
def is_target_suitable_for_task(df, target_column, task_type):
    target_series = df[target_column]
    if task_type == "Regression":
        return pd.api.types.is_numeric_dtype(target_series) or target_series.dtype == 'int64' or target_series.dtype == 'float64'
    elif task_type == "Classification":
        # Check if target is categorical or has limited unique values (e.g., binary or small number of classes)
        unique_values = target_series.nunique()
        return pd.api.types.is_object_dtype(target_series) or unique_values < 20  # Arbitrary threshold for classification
    return False

# Sidebar for Dataset Upload and Inputs
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # Save the dataset for the generated code to use
        with open("uploaded_dataset.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("✅ File Uploaded!")
        
        # Dataset Preview
        st.subheader("📊 Dataset Preview")
        st.write(df.head())

        # Input Selection
        target_column = st.sidebar.selectbox("🎯 Select Target Variable", df.columns)
        feature_columns = st.sidebar.multiselect("🔍 Select Feature Columns", [col for col in df.columns if col != target_column])
        task_type = st.sidebar.radio("📈 Task Type", ["Regression", "Classification"])
        
        # Check dataset suitability
        if not is_target_suitable_for_task(df, target_column, task_type):
            if task_type == "Regression":
                st.warning("⚠️ Warning: The selected target variable is categorical or has too many unique values, making it unsuitable for regression. Consider using a classification task instead.")
            else:  # Classification
                st.warning("⚠️ Warning: The selected target variable is continuous, making it unsuitable for classification. Consider using a regression task instead.")
        else:
            st.sidebar.success("✅ Target variable is suitable for the selected task.")

        model_options = {
            "Regression": ["Linear Regression", "Random Forest Regressor"],
            "Classification": ["Logistic Regression", "Random Forest Classifier"]
        }
        model_type = st.sidebar.selectbox("🤖 Model Type", model_options[task_type])

        # Train Model Button
        if st.sidebar.button("🚀 Train Model"):
            if not feature_columns:
                st.error("❌ Please select at least one feature column.")
            else:
                with st.spinner("Training Model..."):
                    # Preprocess Data
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()

                    # Encode Categorical Variables
                    le = LabelEncoder()
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            X[col] = le.fit_transform(X[col].astype(str))
                    if y.dtype == 'object' and task_type == "Classification":
                        y = le.fit_transform(y.astype(str))
                    elif y.dtype == 'object' and task_type == "Regression":
                        st.warning("⚠️ Converting categorical target to numeric for regression (not ideal). Consider reselecting task type.")
                        y = le.fit_transform(y.astype(str))

                    # Convert to numeric and handle missing values
                    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                    y = pd.Series(y).apply(pd.to_numeric, errors='coerce').fillna(0)

                    # Scale Features
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)

                    # Split Data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train Model
                    model = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Random Forest Classifier": RandomForestClassifier(n_estimators=100)
                    }[model_type]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Model Performance
                    st.subheader("📊 Model Performance")
                    if task_type == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.write(f"🔹 Mean Squared Error: **{mse:.4f}**")
                        st.write(f"🔹 R² Score: **{r2:.4f}**")

                        # Residual Plot
                        try:
                            residuals = y_test - y_pred
                            # Ensure y_test and residuals are numeric and 1D
                            y_test = np.array(y_test).flatten()
                            residuals = np.array(residuals).flatten()
                            # Check for non-numeric values
                            if not np.issubdtype(y_test.dtype, np.number) or not np.issubdtype(residuals.dtype, np.number):
                                raise ValueError("y_test or residuals contain non-numeric values")
                            fig = px.scatter(x=y_test, y=residuals, title="Residual Analysis",
                                            labels={'x': 'Actual Values', 'y': 'Residuals'})
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"❌ Error generating residual plot: {str(e)}")

                    else:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"🔹 Accuracy: **{accuracy:.4f}**")
                        st.text("🔹 Classification Report:")
                        st.text(classification_report(y_test, y_pred))

                        # Confusion Matrix
                        try:
                            cm = confusion_matrix(y_test, y_pred)
                            fig = ff.create_annotated_heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0])),
                                                            annotation_text=cm.astype(str), colorscale='Blues')
                            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"❌ Error generating confusion matrix: {str(e)}")

                    # Generate Code
                    st.subheader("📝 AI-Generated Code")
                    code = generate_code(task_type, model_type, feature_columns, target_column)
                    if "⚠️" in code:
                        st.error(f"Failed to generate code: {code}")
                    else:
                        st.code(code, language="python")

    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
    finally:
        # Clean up the saved dataset file
        if os.path.exists("uploaded_dataset.csv"):
            os.remove("uploaded_dataset.csv")
else:
    st.info("📂 Upload a dataset to get started!")