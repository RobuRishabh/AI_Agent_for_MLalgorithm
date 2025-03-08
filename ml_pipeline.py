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
        # Explicitly checks for int64 or float64 types, ensuring compatibility with regression (e.g., price in Diamonds is float64).
        return pd.api.types.is_numeric_dtype(target_series) or target_series.dtype == 'int64' or target_series.dtype == 'float64' 
    elif task_type == "Classification":
        unique_values = target_series.nunique() # Count unique values
        return pd.api.types.is_object_dtype(target_series) or unique_values < 20 # Allows multi-classification if the target has fewer than 20 unique values
    return False

def preprocess_data(df, feature_columns, target_column, task_type):
    """Preprocess the dataset for ML training."""
    X = df[feature_columns].copy() # Independent Variables - predictor columns
    y = df[target_column].copy() # Target Variable - column to predict

    # Encode Categorical Variables
    le = LabelEncoder()
    # Iterate over each column in X and y
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))
    if y.dtype == 'object' and task_type == "Classification": # check if the target is categorical and task is classification
        # Encode categorical target for classification
        y = le.fit_transform(y.astype(str))
    elif y.dtype == 'object' and task_type == "Regression": # check if the target is categorical and task is regression
        # Convert categorical target to numeric for regression
        y = le.fit_transform(y.astype(str))
        return X, y, "⚠️ Converting categorical target to numeric for regression (not ideal). Consider reselecting task type."

    # Converts all columns in X and y to numeric types, setting non-convertible values to NaN.
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.Series(y).apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale Features
    scaler = StandardScaler() # Initializes a scaler to standardize features (mean = 0, variance = 1).
    X = scaler.fit_transform(X) # Fits the scaler to X and transforms it.

    return X, y, None # Returns the preprocessed X (scaled features), y (target), and None (no warning).

def train_and_evaluate(X, y, task_type, model_type):
    """Train the model and evaluate its performance."""
    # Splits data into 80% training and 20% test sets, with a fixed random state for reproducibility.
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
            # Computes the difference between actual and predicted values.
            residuals = y_test - y_pred
            # Converts y_test to a 1D NumPy array.
            y_test = np.array(y_test).flatten()
            # Ensures residuals are 1D.
            residuals = np.array(residuals).flatten()
            if not np.issubdtype(y_test.dtype, np.number) or not np.issubdtype(residuals.dtype, np.number):
                raise ValueError("y_test or residuals contain non-numeric values")
            # Creates a scatter plot of actual values vs. residuals.
            fig = px.scatter(x=y_test, y=residuals, title="Residual Analysis",
                            labels={'x': 'Actual Values', 'y': 'Residuals'})
            # Adds a horizontal line at y=0 to indicate ideal residuals.
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            # Returns metrics, the plot figure, and no error.
            return metrics, fig, None
        except Exception as e:
            return metrics, None, f"❌ Error generating residual plot: {str(e)}"

    else:
        accuracy = accuracy_score(y_test, y_pred) # Calculates the proportion of correct predictions.
        report = classification_report(y_test, y_pred) # Generates a detailed report (precision, recall, F1-score).
        metrics = {"accuracy": accuracy, "report": report} # Stores accuracy and report in a dictionary.

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0])),
                                              annotation_text=cm.astype(str), colorscale='Blues')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            return metrics, fig, None
        except Exception as e:
            return metrics, None, f"❌ Error generating confusion matrix: {str(e)}"