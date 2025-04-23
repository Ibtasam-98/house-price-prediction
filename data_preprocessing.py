import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """Loading the dataset and preprocesses it for modeling."""
    df = pd.read_csv(file_path)
    df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)
    X = df.drop('price', axis=1)
    y = df['price']
    y_binary = (y > y.median()).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_binary, y.median(), scaler, X.columns

if __name__ == "__main__":
    X_scaled, y_binary, median_price, scaler, feature_names = load_and_preprocess_data("dataset/housing.csv")
    print("Scaled Features (first 5 rows):\n", X_scaled[:5])
    print("\nBinary Target (first 5 values):\n", y_binary[:5])
    print("\nMedian Price:", median_price)
    print("\nScaler object:", scaler)
    print("\nFeature Names:", feature_names)