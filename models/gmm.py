import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class SentimentGMM:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)

    def fit(self, X):
        self.gmm.fit(X)

    def predict(self, X):
        return self.gmm.predict(X)

    def evaluate(self, X):
        labels = self.gmm.predict(X)
        return silhouette_score(X, labels)

def load_data(X_file, y_file=None):
    X = np.load(X_file)
    y_data = np.load(y_file)
    y = y_data['single'] 
    return X, y

def main():
    # Load data
    X, y = load_data('../data/SEntFiN_dataset/X_4000.npy', '../data/SEntFiN_dataset/y.npz')

    # Initialize and fit the model
    model = SentimentGMM()
    model.fit(X)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate the model
    silhouette_avg = model.evaluate(X)
    print(f"Silhouette Score: {silhouette_avg}")

    # Map GMM components to sentiment labels
    unique_components = np.unique(predictions)
    component_means = [np.mean(y[predictions == component]) for component in unique_components]
    sentiment_mapping = {
        component: 'Negative' if mean <= -0.33 else 'Positive' if mean >= 0.33 else 'Neutral'
        for component, mean in zip(unique_components, component_means)
    }
    
    sentiment_predictions = np.array([sentiment_mapping[label] for label in predictions])

    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(X))):
        print(f"Sample {i}: Predicted Component: {predictions[i]}, Predicted Sentiment: {sentiment_predictions[i]}, True Sentiment: {y[i]}")

    # Calculate accuracy
    true_sentiments = np.array(['Negative' if s <= -0.33 else 'Positive' if s >= 0.33 else 'Neutral' for s in y])
    accuracy = np.mean(sentiment_predictions == true_sentiments)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Save results
    np.savez('gmm_results.npz', X=X, true_sentiment=y, predicted_component=predictions, predicted_sentiment=sentiment_predictions)

if __name__ == "__main__":
    main()