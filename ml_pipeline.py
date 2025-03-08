import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

def is_target_suitable_for_task(df, target_column, task_type):
    """Check if the target variable is suitable for the selected task."""
    target_series = df[target_column]
    if task_type == "Regression":
        return pd.api.types.is_numeric_dtype(target_series) or target_series.dtype == 'int64' or target_series.dtype == 'float64'
    elif task_type == "Classification":
        unique_values = target_series.nunique()
        return pd.api.types.is_object_dtype(target_series) or unique_values < 20
    return False

def preprocess_data(df, feature_columns, target_column, task_type):
    """Preprocess the dataset for ML training."""
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
        y = le.fit_transform(y.astype(str))
        return X, y, "⚠️ Converting categorical target to numeric for regression (not ideal). Consider reselecting task type."

    # Convert to numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.Series(y).apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, None

def train_and_evaluate(X, y, task_type, model_type):
    """Train the model and evaluate its performance."""
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

    # Evaluate Model
    if task_type == "Regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"mse": mse, "r2": r2}

        # Residual Plot
        try:
            residuals = y_test - y_pred
            y_test = np.array(y_test).flatten()
            residuals = np.array(residuals).flatten()
            if not np.issubdtype(y_test.dtype, np.number) or not np.issubdtype(residuals.dtype, np.number):
                raise ValueError("y_test or residuals contain non-numeric values")
            fig = px.scatter(x=y_test, y=residuals, title="Residual Analysis",
                            labels={'x': 'Actual Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            return metrics, fig, None
        except Exception as e:
            return metrics, None, f"❌ Error generating residual plot: {str(e)}"

    else:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        metrics = {"accuracy": accuracy, "report": report}

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0])),
                                              annotation_text=cm.astype(str), colorscale='Blues')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            return metrics, fig, None
        except Exception as e:
            return metrics, None, f"❌ Error generating confusion matrix: {str(e)}"