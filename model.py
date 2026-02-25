import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset (use your correct path)
data = pd.read_csv(
    r"C:\Users\admin\Downloads\house-prices-advanced-regression-techniques\train.csv"
)

# Select features
features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "HalfBath",
    "Neighborhood"
]

data = data[features + ["SalePrice"]].dropna()

# One-hot encode location
data_encoded = pd.get_dummies(data, columns=["Neighborhood"], drop_first=True)

X = data_encoded.drop("SalePrice", axis=1)
y = data_encoded["SalePrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model + feature names
with open("house_price_model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("✅ Model trained using square footage + location")