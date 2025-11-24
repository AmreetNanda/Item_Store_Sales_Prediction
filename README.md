# Item Store Sales prediction App (Machine Learning + Flask + Docker)

The Item store Sales Prediction Web App is an interactive, user-friendly tool that forecasts sales for products in BigMart outlets based on multiple features. This project demonstrates how a machine learning model can be integrated with a web interface to provide a seamless experience for users who want to estimate sales.

It includes:
- A complete machine learning pipeline
- A Flask web UI for model training 
- A fully modular ML codebase
- Optional Docker container for deployment
---

## Features

- Input Form: Users can enter product details such as weight, visibility, MRP, outlet information, item type, and fat content.
- Machine Learning Prediction: The application predicts sales based on input attributes using a trained machine learning model.
- User-Friendly Interface: Users can submit product and outlet data easily and receive sales forecasts.
- Dataset: A BigMart dataset containing product and outlet features along with sales values. The model is trained to understand the relationship between product characteristics, outlet details, and sales.

## Attributes in the Dataset:
- Item_Weight: Weight of the product.
- Item_Visibility: Visibility of the product in the outlet.
- Item_MRP: Maximum retail price of the product.
- Item_Fat_Content: Fat content category of the product.
- Item_Type: Category of the product.
- Outlet_Size: Size of the outlet (Small, Medium, High).
- Outlet_Location_Type: Location type of the outlet.
- Outlet_Type: Type of the outlet (Grocery Store, Supermarket Type1, etc.)
- Outlet_Establishment_Year: Year in which the outlet was established.
- Outlet_Age: Age of the outlet (calculated from establishment year).
- Item_Identifier_Categories: Category derived from Item_Identifier.

## Technologies Used:
- Front-End: HTML, CSS
- Back-End: Python (Flask framework)
- Machine Learning: Linear Regression, Lasso Regression, Ridge Regression, ElasticNet, Decision Tree Regressor, Random Forest, XGBRegressor

## Project Structure
```bash
Item_Store_Sales_Price_Prediction/ # Modular ML package
│
├── app.py # Flask app
├── EDA
├── Template
│ ├── form.html
│ └── index.html
├── src
│ ├── components
│ │   ├── __init__.py
│ │   ├── data_ingestion.py
│ │   ├── data_transformation.py
│ │   └── model_trainer.py
│ ├── pipeline
│ │   ├── __init__.py
│ │	  ├── predict_pipeline.py
│ │   └── training_pipeline.py
│ ├── Exception.py
│ ├── Logger.py
│ └── Utils.py
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container
├── run.sh # Optional to run the script
└── setup.py # Optional to run the script
```
```
## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Item_Store_Sales_Prediction.git
cd Item_Store_Sales_Price_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Flask app
```bash
run app.py
```
Open in your browser:
👉 http://127.0.0.1:5000/
👉 Enter the attributes of the items in stores in the input form.
👉 Click the "Predict Price" button.
👉 Receive the predicted sales price

## 🐳 Running with Docker (optional)
### Build the image
```bash
docker build -t Item_Store_Sales_Prediction .
```

### Run the container
```bash
docker run -p 8501:8501 Item_Store_Sales_Prediction
```
Open: 👉 http://localhost:8501


## Screenshots
##### Home page
![App Screenshot](https://github.com/AmreetNanda/Item_Store_Sales_Prediction/blob/main/Item_Store_Sales_Prediction_1.png)

##### Form 
![App Screenshot](https://github.com/AmreetNanda/Item_Store_Sales_Prediction/blob/main/Item_Store_Sales_Prediction_2.png)

## Demo
https://github.com/AmreetNanda/Item_Store_Sales_Prediction/blob/main/Item_Sales_Prediction_Demo.mp4

## License
[MIT](https://choosealicense.com/licenses/mit/)