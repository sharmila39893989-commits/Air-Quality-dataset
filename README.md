# 🏠 House Price Prediction using Regression

## 📌 Project Overview

This project predicts house prices using Machine Learning Regression.

The project uses a House Price dataset containing different features related to houses. A Linear Regression model is trained to predict the `House_Price`.

The trained model is saved using Pickle and then used in a Streamlit application for making house price predictions.

---

## 🎯 Project Objective

The main objective of this project is to predict the price of a house based on its features.

The input features used in this project are:

* Square Footage
* Number of Bedrooms
* Number of Bathrooms
* Year Built
* Lot Size
* Garage Size
* Neighborhood Quality

The target variable is:

* House Price

---

## 📂 Dataset

The dataset used for this project is:

`house_price_regression_dataset.csv`

The dataset contains:

* 1000 rows
* 8 columns

### Features

| Feature              | Description                       |
| -------------------- | --------------------------------- |
| Square_Footage       | Size of the house in square feet  |
| Num_Bedrooms         | Number of bedrooms                |
| Num_Bathrooms        | Number of bathrooms               |
| Year_Built           | Year in which the house was built |
| Lot_Size             | Size of the property lot          |
| Garage_Size          | Garage capacity                   |
| Neighborhood_Quality | Quality score of the neighborhood |

### Target

`House_Price`

---

## 🔍 Project Workflow

The project follows these steps:

1. Import required libraries
2. Load the dataset
3. Perform data analysis
4. Check dataset shape
5. Check dataset information
6. Perform descriptive statistical analysis
7. Check missing values
8. Check duplicate values
9. Analyze data types
10. Perform data visualization
11. Perform correlation analysis
12. Separate features and target
13. Split data into training and testing sets
14. Apply Linear Regression
15. Train the model
16. Make predictions
17. Evaluate the model
18. Calculate R² Score
19. Calculate MAE
20. Calculate RMSE
21. Save the trained model using Pickle
22. Create a Streamlit application
23. Deploy the application

---

## 🤖 Machine Learning Algorithm

### Linear Regression

Linear Regression is used because the target variable `House_Price` is a continuous numerical value.

The model learns the relationship between the input features and the house price.

---

## 📊 Model Evaluation

The model is evaluated using:

### R² Score

R² Score measures how well the model explains the variation in the target variable.

The model achieved approximately:

**R² Score: 99.84%**

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted house prices.

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

---

## 💾 Model Saving

The trained Linear Regression model is saved using Pickle.

The saved model file is:

```text
house_price_model.pkl
```

This file is loaded by the Streamlit application to make predictions.

---

## 🌐 Streamlit Application

The project includes a Streamlit web application.

The application allows the user to enter:

* Square Footage
* Number of Bedrooms
* Number of Bathrooms
* Year Built
* Lot Size
* Garage Size
* Neighborhood Quality

After clicking the **Predict House Price** button, the application displays the estimated house price.

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd House_Price_Prediction
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── app.py
├── house_price_model.pkl
├── house_price_regression_dataset.csv
├── House_Price_Regression.ipynb
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pickle
* Streamlit
* GitHub

---

## 📌 Conclusion

This project demonstrates how Regression Machine Learning can be used to predict house prices.

The dataset was analyzed, the Linear Regression model was trained and evaluated, and the trained model was saved using Pickle.

Finally, a Streamlit web application was developed to allow users to enter house details and receive a predicted house price.

---

## 👩‍💻 Project

**House Price Prediction using Regression**

Machine Learning Regression Project

```
```
