import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SentimentDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance

    def fit(self, X):
        self.X_transformed = self.pca.fit_transform(self.scaler.fit_transform(X))
        self.dbscan.fit(self.X_transformed)

    def predict(self, X):
        X_transformed = self.pca.transform(self.scaler.transform(X))
        return self.dbscan.fit_predict(X_transformed)

    def evaluate(self, X):
        labels = self.dbscan.labels_
        mask = labels != -1
        if np.sum(mask) > 1:
            return silhouette_score(self.X_transformed[mask], labels[mask])
        else:
            return -1

def load_data(X_file, y_file=None):
    X = np.load(X_file)
    y_data = np.load(y_file)
    y = y_data['single'] 
    return X, y

def main():
    X, y = load_data('../data/SEntFiN_dataset/X_4000.npy', '../data/SEntFiN_dataset/y.npz')

    # Grid search for DBSCAN parameters
    best_score = -np.inf
    best_params = None
    best_predictions = None

    for eps in np.linspace(0.1, 2, 20):
        for min_samples in range(2, 11):
            model = SentimentDBSCAN(eps=eps, min_samples=min_samples)
            model.fit(X)
            predictions = model.predict(X)
            silhouette_avg = model.evaluate(X)

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_params = (eps, min_samples)
                best_predictions = predictions

    print(f"Best parameters: eps={best_params[0]}, min_samples={best_params[1]}")
    print(f"Best Silhouette Score: {best_score}")

    # Use best parameters for final model
    final_model = SentimentDBSCAN(eps=best_params[0], min_samples=best_params[1])
    final_model.fit(X)
    predictions = final_model.predict(X)

    # Map DBSCAN clusters to sentiment labels
    unique_clusters = np.unique(predictions)
    cluster_means = [np.mean(y[predictions == cluster]) for cluster in unique_clusters if cluster != -1]
    sentiment_mapping = {
        cluster: 'Negative' if mean <= -0.33 else 'Positive' if mean >= 0.33 else 'Neutral'
        for cluster, mean in zip(unique_clusters, cluster_means) if cluster != -1
    }
    sentiment_mapping[-1] = 'Neutral'  # Assign noise points to 'Neutral'
    
    sentiment_predictions = np.array([sentiment_mapping[label] for label in predictions])

    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(X))):
        print(f"Sample {i}: Predicted Cluster: {predictions[i]}, Predicted Sentiment: {sentiment_predictions[i]}, True Sentiment: {y[i]}")

    # Calculate accuracy
    true_sentiments = np.array(['Negative' if s <= -0.33 else 'Positive' if s >= 0.33 else 'Neutral' for s in y])
    accuracy = np.mean(sentiment_predictions == true_sentiments)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Save results
    np.savez('dbscan_results.npz', X=X, true_sentiment=y, predicted_cluster=predictions, predicted_sentiment=sentiment_predictions)

if __name__ == "__main__":
    main()