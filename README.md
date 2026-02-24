🏠 House Price Prediction Web App (Streamlit)
📌 Project Overview

This project is a House Price Prediction Web Application built using Machine Learning and Streamlit.
The application predicts the selling price of a house based on key features such as:

Square Footage

Number of Bedrooms

Number of Full Bathrooms

Number of Half Bathrooms

The prediction model is trained using the Kaggle House Prices: Advanced Regression Techniques dataset and deployed as an interactive web application using Streamlit.

🎯 Objectives

Build a Linear Regression model for house price prediction

Understand feature selection and data preprocessing

Create an interactive web app for real-time predictions

Gain hands-on experience in ML model deployment

🧠 Machine Learning Model

Algorithm: Linear Regression

Library Used: scikit-learn

Target Variable: SalePrice

Input Features:

GrLivArea (Above ground living area in square feet)

BedroomAbvGr (Number of bedrooms)

FullBath (Number of full bathrooms)

HalfBath (Number of half bathrooms)

Missing values are handled using mean imputation.

🛠️ Technologies Used

Python

Pandas & NumPy – Data processing

Scikit-learn – Model training

Streamlit – Web application framework

Pickle – Model serialization

📁 Project Structure
internship-projects/
│
├── train.csv                  # Kaggle dataset
├── model.py                   # Model training script
├── house_price_model.pkl      # Trained model
├── app_streamlit.py           # Streamlit web app
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

▶️ How to Run the Project
1️⃣ Install Required Libraries
pip install -r requirements.txt

Or manually:

pip install pandas numpy scikit-learn streamlit
2️⃣ Train the Model
python model.py

This will generate the trained model file:

house_price_model.pkl
3️⃣ Run the Streamlit App
streamlit run app_streamlit.py

The application will open in your browser at:

http://localhost:8501
📊 Application Features

User-friendly sidebar inputs

Real-time house price prediction

Interactive web interface

Simple and fast model inference

🧪 Sample Input

Square Footage: 900

Bedrooms: 2

Full Bathrooms: 2

Half Bathrooms: 1

💡 Sample Output
Estimated House Price: ₹ 1,45,000

(Note: Price depends on model training and dataset distribution)

🚀 Future Enhancements

Add more features for higher accuracy

Use advanced models (Ridge, Lasso, Random Forest)

Add data visualizations

Deploy app online using Streamlit Cloud or Render

Improve UI with charts and insights

📚 Dataset

Source: Kaggle – House Prices: Advanced Regression Techniques

Link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

👤 Author

Harsha Reddy
Internship / Academic Machine Learning Project

⭐ Acknowledgements

Kaggle for providing the dataset

Streamlit for the simple and powerful deployment framework
