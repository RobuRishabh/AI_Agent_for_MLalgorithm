import streamlit as st
import pandas as pd
from ml_pipeline import is_target_suitable_for_task, preprocess_data, train_and_evaluate
from code_generator import generate_code
import os

# Streamlit UI Setup
st.set_page_config(page_title="AI ML Agent", layout="wide")
st.title("🤖 Code Generation ML Agent with Ollama ")

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
            else:
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
                    X, y, warning = preprocess_data(df, feature_columns, target_column, task_type)
                    if warning:
                        st.warning(warning)

                    # Train and Evaluate
                    metrics, fig, error = train_and_evaluate(X, y, task_type, model_type)

                    # Display Results
                    st.subheader("📊 Model Performance")
                    if task_type == "Regression":
                        st.write(f"🔹 Mean Squared Error: **{metrics['mse']:.4f}**")
                        st.write(f"🔹 R² Score: **{metrics['r2']:.4f}**")
                        if fig:
                            st.plotly_chart(fig)
                        if error:
                            st.error(error)
                    else:
                        st.write(f"🔹 Accuracy: **{metrics['accuracy']:.4f}**")
                        st.text("🔹 Classification Report:")
                        st.text(metrics['report'])
                        if fig:
                            st.plotly_chart(fig)
                        if error:
                            st.error(error)

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