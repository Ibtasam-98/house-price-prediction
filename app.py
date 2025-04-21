import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from flask import Flask, request, jsonify
from scipy import stats
import joblib

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv("housing.csv")

# Outlier Handling (Remove outliers based on z-score)
z_scores = np.abs(stats.zscore(df['price']))
df_filtered = df[(z_scores < 3)]
X = df_filtered.drop('price', axis=1)
y = df_filtered['price']

X = pd.get_dummies(X, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# Train the RandomForestRegressor model (with hyperparameter tuning)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Save the model, scaler, and polynomial features
joblib.dump(best_model, 'housing_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')

# Load the model, scaler, and polynomial features
loaded_model = joblib.load('housing_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_poly = joblib.load('poly.pkl')

# Prediction function (using the loaded model)
def predict_price(area, bedrooms, bathrooms, stories, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, parking, prefarea_yes, furnishingstatus_semi, furnishingstatus_unfurnished):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, parking, prefarea_yes, furnishingstatus_semi, furnishingstatus_unfurnished]], columns=X.columns)
    input_scaled = loaded_scaler.transform(input_data)
    input_poly = loaded_poly.transform(input_scaled)
    predicted_price = loaded_model.predict(input_poly)
    return predicted_price[0]

# Classification function (Expensive or Not Expensive)
def classify_price(predicted_price, median_price):
    if predicted_price > median_price:
        return "Expensive"
    else:
        return "Not Expensive"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        stories = int(data['stories'])
        mainroad_yes = int(data['mainroad_yes'])
        guestroom_yes = int(data['guestroom_yes'])
        basement_yes = int(data['basement_yes'])
        hotwaterheating_yes = int(data['hotwaterheating_yes'])
        airconditioning_yes = int(data['airconditioning_yes'])
        parking = int(data['parking'])
        prefarea_yes = int(data['prefarea_yes'])
        furnishing_choice = int(data['furnishing_choice'])

        if furnishing_choice == 1:
            furnishingstatus_semi = 1
            furnishingstatus_unfurnished = 0
        elif furnishing_choice == 2:
            furnishingstatus_semi = 0
            furnishingstatus_unfurnished = 1
        else:
            furnishingstatus_semi = 0
            furnishingstatus_unfurnished = 0

        predicted_price = predict_price(area, bedrooms, bathrooms, stories, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, parking, prefarea_yes, furnishingstatus_semi, furnishingstatus_unfurnished)

        # Calculate median price for classification
        median_price = y.median()
        price_classification = classify_price(predicted_price, median_price)

        return jsonify({'predicted_price': predicted_price, 'price_classification': price_classification})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)