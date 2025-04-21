from data_preprocessing import load_and_preprocess_data
from data_visualization import visualize_data
from model_training import train_and_evaluate_model
from prediction import main as prediction_main
import pandas as pd

if __name__ == "__main__":
    # Load and preprocess data
    X_scaled, y_binary, _, _, _ = load_and_preprocess_data("dataset/housing.csv")

    # Load the original dataframe for visualization
    df_original = pd.read_csv("dataset/housing.csv")
    df_processed = pd.get_dummies(df_original, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

    # Visualize data
    visualize_data(df_processed)

    # Train and evaluate the model
    train_and_evaluate_model(X_scaled, y_binary)

    # Run the prediction loop
    prediction_main()


