from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_and_evaluate_model(X, y):
    """Trains a Random Forest model, evaluates it, and visualizes results."""
    # Polynomial Features
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Model Evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Expensive', 'Expensive'],
                yticklabels=['Not Expensive', 'Expensive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Visualize classification report
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['Not Expensive', 'Expensive']
    values = [[report['0'][metric], report['1'][metric]] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [v[0] for v in values], width, label=classes[0])
    rects2 = ax.bar(x + width/2, [v[1] for v in values], width, label=classes[1])

    ax.set_ylabel('Scores')
    ax.set_title('Classification Report')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()
    plt.savefig("classification_report.png")
    plt.show()

    return model, poly

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    X_scaled, y_binary, _, _, _ = load_and_preprocess_data("housing.csv")
    trained_model, polynomial_features = train_and_evaluate_model(X_scaled, y_binary)
    print("\nTrained Random Forest Model:", trained_model)
    print("\nPolynomial Features Object:", polynomial_features)