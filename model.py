import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset (use YOUR correct path if needed)
data = pd.read_csv(
    r"C:\Users\admin\Downloads\house-prices-advanced-regression-techniques\train.csv"
)

# Features
features = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
X = data[features].fillna(data[features].mean())
y = data["SalePrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")