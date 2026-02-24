from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sqft = float(request.form["sqft"])
    bedrooms = int(request.form["bedrooms"])
    full_bath = int(request.form["full_bath"])
    half_bath = int(request.form["half_bath"])

    features = np.array([[sqft, bedrooms, full_bath, half_bath]])
    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price: ₹ {round(prediction, 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)