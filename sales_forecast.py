# Sales Forecasting using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: 30 days revenue
data = {
    'date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'revenue': np.random.randint(1000, 5000, size=30)
}
df = pd.DataFrame(data)

# Extract date features
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

X = df[['day', 'month', 'year']]
y = df['revenue']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save prediction to CSV 
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_csv("predictions.csv",
index=False)

# Plot graph
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Sample')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.savefig("sales_prediction.png")
plt.show()