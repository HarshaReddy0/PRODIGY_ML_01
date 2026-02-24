import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample data mimicking Indian house prices (₹, sq ft) from real dataset stats[web:1]
# In practice, load from CSV: pd.read_csv('house_price.csv')
data = {
    'area': [7420, 8960, 9960, 7500, 7420, 1650, 4500, 7200, 5151, 16200],
    'bedrooms': [4, 4, 3, 4, 4, 2, 3, 4, 3, 5],
    'bathrooms': [2, 4, 2, 2, 1, 1, 2, 3, 2, 4],
    'price': [13300000, 12250000, 12250000, 12215000, 11410000, 1750000, 4760000, 7200000, 4340000, 13300000]  # ₹
}
df = pd.DataFrame(data)

# Features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (recommended for better performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {np.sqrt(mse):,.0f} ₹")
print(f"R²: {r2:.3f}")
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept:", model.intercept_)
