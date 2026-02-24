import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(
    r"C:\Users\admin\Downloads\house-prices-advanced-regression-techniques"
)

# Select features
features = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
X = data[features]
y = data["SalePrice"]

# Handle missing values
X = X.fillna(X.mean())

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")