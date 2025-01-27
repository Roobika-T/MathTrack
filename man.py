!pip install pandas matplotlib scikit-learn statsmodels gradio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import CubicSpline
import numpy as np
import gradio as gr

# Load data from CSV
def load_data(filepath):
    return pd.read_csv(filepath)

# Summarize expenses by category
def summarize_expenses(data):
    if 'Transaction Type' not in data.columns:
        return "Error: 'Transaction Type' column not found in the data."
    return data[data['Transaction Type'] == 'debit'].groupby('Category')['Amount'].sum()

# Train a Naive Bayes classifier for categorization
def train_naive_bayes(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Description'])
    y = data['Category']
    clf = MultinomialNB()
    clf.fit(X, y)
    return clf, vectorizer

# Newton-Raphson method to optimize budget allocation
def newton_raphson(expense_summary, budget_amount):
    def f(x):
        return expense_summary - budget_amount * x
    def f_prime(x):
        return -budget_amount
    # Initial guess
    x0 = 1.0
    epsilon = 1e-6
    for _ in range(10):
        x0 = x0 - f(x0) / f_prime(x0)
    return x0

# Gauss-Seidel method to approximate budget overshoot
def gauss_seidel(expense_summary, budget_amount):
    if budget_amount <= 0:
        raise ValueError("Budget amount must be greater than zero.")

    x = expense_summary / budget_amount
    for i in range(100):  # Iteration limit
        x_new = (expense_summary - budget_amount * x) / budget_amount
        if np.allclose(x_new, x, rtol=1e-5):
            break
        x = x_new
    return x

# Power Method for dominant spending category
def power_method(expense_summary):
    if expense_summary.empty:
        raise ValueError("Expense summary is empty; unable to perform Power Method.")

    # Create a diagonal matrix from the expense summary
    matrix_size = len(expense_summary)
    if matrix_size == 0:
        raise ValueError("Expense summary is empty; unable to perform Power Method.")

    matrix = np.diag(expense_summary.values)  # Create a diagonal matrix
    b = np.random.rand(matrix_size)  # Random initial vector

    for _ in range(10):  # Iteration limit
        b = np.dot(matrix, b)
        b = b / np.linalg.norm(b)  # Normalize the vector

    # Dominant value
    dominant_value = np.max(b)
    return pd.Series([dominant_value], index=[expense_summary.index[0]])  # Returning as Series for consistency


# Forecast future expenses using ARIMA
def forecast_expenses(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    monthly_expenses = data['Amount'].resample('M').sum()

    model = ARIMA(monthly_expenses, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)

    return forecast

# Cubic Spline Curve Fitting for forecasting
def spline_forecasting(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Resampling monthly expenses
    monthly_expenses = data['Amount'].resample('M').sum()

    # Debugging: Check the resampled data
    print("Monthly Expenses for Spline Forecasting:")
    print(monthly_expenses)

    if len(monthly_expenses) < 3:
        raise ValueError("Not enough data points for cubic spline interpolation. Minimum is 3.")

    # Using Cubic Spline for smoother forecast
    x = np.arange(len(monthly_expenses))
    spline = CubicSpline(x, monthly_expenses.values)
    forecast_x = np.arange(len(monthly_expenses), len(monthly_expenses) + 3)
    forecast = spline(forecast_x)

    # Ensure the forecast output is a valid array
    if len(forecast) != 3:
        raise ValueError("Forecast did not return the expected number of points.")

    return pd.Series(forecast, index=pd.date_range(start=monthly_expenses.index[-1] + pd.DateOffset(months=1), periods=3, freq='M'))


# Generate plot for future expenses
def plot_forecast(forecast, method_name):
    plt.figure(figsize=(10, 5))
    plt.plot(forecast.index, forecast, marker='o', color='b')
    plt.title(f"Forecasted Future Expenses ({method_name})")
    plt.xlabel("Date")
    plt.ylabel("Predicted Amount")
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'forecast_{method_name}.png')
    plt.close()

# Predict category for a new transaction
def predict_category(clf, vectorizer, description):
    X_new = vectorizer.transform([description])
    predicted = clf.predict(X_new)
    return predicted[0]

# Main function to process input and return results
def main(file, description, budget_category, budget_amount, algorithm_choice):
    try:
        # Load data
        data = load_data(file.name)

        # Summarize expenses
        expense_summary = summarize_expenses(data)

        # Train Naive Bayes classifier
        clf, vectorizer = train_naive_bayes(data)

        # Predict category for new transaction
        predicted_category = predict_category(clf, vectorizer, description)

        # Forecast future expenses based on chosen algorithm
        if algorithm_choice == "ARIMA":
            future_expenses = forecast_expenses(data)
            method_name = "ARIMA"
        elif algorithm_choice == "Spline":
            future_expenses = spline_forecasting(data)
            method_name = "Cubic Spline"
        elif algorithm_choice == "Newton-Raphson":
            budget_optimized = newton_raphson(expense_summary, budget_amount)
            future_expenses = budget_optimized  # You can plot this as well
            method_name = "Newton-Raphson"
        elif algorithm_choice == "Gauss-Seidel":
            budget_optimized = gauss_seidel(expense_summary, budget_amount)
            future_expenses = budget_optimized
            method_name = "Gauss-Seidel"
        elif algorithm_choice == "Power Method":
            dominant_category = power_method(expense_summary)
            future_expenses = dominant_category
            method_name = "Power Method"

        # Generate plot
        plot_forecast(future_expenses, method_name)

        # Calculate budget status
        spent = expense_summary.get(budget_category, 0)
        budget_status = "Under Budget" if spent <= budget_amount else "Over Budget"

        return (
            expense_summary.to_string(),
            predicted_category,
            future_expenses.to_string(),
            {"Spent": spent, "Budget": budget_amount, "Status": budget_status},
            f'forecast_{method_name}.png'  # Return the path to the forecast image
        )

    except Exception as e:
        return f"Error: {str(e)}", "", "", {}, ""

# Gradio interface
# Modified Gradio interface with Dropdown for Budget Category
iface = gr.Interface(
    fn=main,
    inputs=[
        gr.File(label="Upload Transaction CSV"),
        gr.Textbox(label="Description for Prediction"),
        gr.Dropdown(label="Budget Category", choices=["Food", "Transport", "Entertainment", "Health", "Other"]),  # Replace with actual categories
        gr.Number(label="Budget Amount"),
        gr.Radio(label="Choose Forecasting Algorithm", choices=["ARIMA", "Spline", "Newton-Raphson", "Gauss-Seidel", "Power Method"])
    ],
    outputs=[
        gr.Textbox(label="Expense Summary"),
        gr.Textbox(label="Predicted Category"),
        gr.Textbox(label="Forecasted Future Expenses"),
        gr.JSON(label="Budget Summary"),
        gr.Image(label="Forecast Plot")
    ],
    title="Advanced Personal Finance Tracker",
    description="Upload your transaction data, get expense summaries, category predictions, and forecast future expenses using different algorithms."
)

# Launch the Gradio interface
iface.launch()
