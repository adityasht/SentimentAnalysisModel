import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import extract_features, standardize

class SentimentSVM:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.svm = SVC(kernel=self.kernel)
        self.scaler = None

    def preprocess(self, X):
        X_features, _, _, _ = extract_features(X)
        X_standardized, scaler = standardize(X_features)
        self.scaler = scaler
        return X_standardized

    def fit(self, X, y):
        X_preprocessed = self.preprocess(X)
        self.svm.fit(X_preprocessed, y)

    def predict(self, X):
        X_features, _, _, _ = extract_features(X)
        X_standardized = self.scaler.transform(X_features)
        return self.svm.predict(X_standardized)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df['Title'].values
    y = df['Decisions'].values

    return X, y

def main():
    # Load and preprocess data
    X, y = load_data('SEntFiN.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit SVM model
    model = SentimentSVM(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy, report = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()