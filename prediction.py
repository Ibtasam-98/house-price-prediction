import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict_price_category(model, poly, scaler, feature_columns, median_price,
                            area, bedrooms, bathrooms, stories, mainroad_yes,
                            guestroom_yes, basement_yes, hotwaterheating_yes,
                            airconditioning_yes, parking, prefarea_yes,
                            furnishingstatus_semi, furnishingstatus_unfurnished):
    """Predicts the price category for given input features."""
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, parking, prefarea_yes, furnishingstatus_semi, furnishingstatus_unfurnished]], columns=feature_columns)
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)
    predicted_category = model.predict(input_poly)
    predicted_price = predicted_category[0] * median_price + (1 - predicted_category[0]) * (median_price - 1)
    return predicted_category[0], predicted_price

def main():
    from data_preprocessing import load_and_preprocess_data
    from model_training import train_and_evaluate_model

    X_scaled, y_binary, median_price, scaler, feature_names = load_and_preprocess_data("dataset/housing.csv")
    trained_model, polynomial_features = train_and_evaluate_model(X_scaled, y_binary)

    predictions = []
    input_numbers = []
    predicted_categories = []
    count = 1

    while True:
        try:
            area = float(input("Enter area: "))
            bedrooms = int(input("Enter number of bedrooms: "))
            bathrooms = int(input("Enter number of bathrooms: "))
            stories = int(input("Enter number of stories: "))
            mainroad_yes = int(input("Is it on the main road? (1 for yes, 0 for no): "))
            guestroom_yes = int(input("Does it have a guest room? (1 for yes, 0 for no): "))
            basement_yes = int(input("Does it have a basement? (1 for yes, 0 for no): "))
            hotwaterheating_yes = int(input("Does it have hot water heating? (1 for yes, 0 for no): "))
            airconditioning_yes = int(input("Does it have air conditioning? (1 for yes, 0 for no): "))
            parking = int(input("Enter number of parking spots: "))
            prefarea_yes = int(input("Is it in a preferred area? (1 for yes, 0 for no): "))
            furnishing_choice = int(input("Furnishing status (0: furnished, 1: semi-furnished, 2: unfurnished): "))
            furnishingstatus_semi = 1 if furnishing_choice == 1 else 0
            furnishingstatus_unfurnished = 1 if furnishing_choice == 2 else 0

            predicted_category, predicted_price = predict_price_category(
                trained_model, polynomial_features, scaler, feature_names, median_price,
                area, bedrooms, bathrooms, stories, mainroad_yes, guestroom_yes,
                basement_yes, hotwaterheating_yes, airconditioning_yes, parking,
                prefarea_yes, furnishingstatus_semi, furnishingstatus_unfurnished
            )

            print(f"Predicted Price Category (1=Expensive, 0=Not Expensive): {predicted_category}")
            print(f"Predicted Price: {predicted_price}\n")

            predictions.append(predicted_price)
            input_numbers.append(count)
            predicted_categories.append(predicted_category)
            count += 1
            continue_prediction = input("Do you want to make another prediction? (yes/no): ").lower()
            if continue_prediction != "yes":
                break
        except ValueError:
            print("Invalid input. Please enter valid numerical values.")

    # Visualize predictions
    if predictions:
        plt.figure(figsize=(10, 6))
        plt.plot(input_numbers, predictions, marker='o', linestyle='-', label='Predicted Price')
        plt.plot(input_numbers, [median_price if cat == 1 else (median_price - 1) for cat in predicted_categories],
                 marker='x', linestyle='--', label='Actual Price Category Median')
        plt.xlabel("Prediction Number")
        plt.ylabel("Price")
        plt.title("Actual vs. Predicted House Prices (Line Chart)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()



