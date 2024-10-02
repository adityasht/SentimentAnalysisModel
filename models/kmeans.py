import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class SentimentKMeans:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X):
        self.kmeans.fit(X)

    def predict(self, X):
        return self.kmeans.predict(X)

    def evaluate(self, X):
        labels = self.kmeans.labels_
        return silhouette_score(X, labels)

def load_data(X_file, y_file=None):
    X = np.load(X_file)
    y_data = np.load(y_file)
    y = y_data['single'] 
    return X, y


def main():

    X, y = load_data('../data\SEntFiN_dataset\X_full.npy', '../data/SEntFiN_dataset/y.npz')


    model = SentimentKMeans()
    model.fit(X)

 
    predictions = model.predict(X)


    silhouette_avg = model.evaluate(X)
    print(f"Silhouette Score: {silhouette_avg}")


    unique_clusters = np.unique(predictions)
    cluster_means = [np.mean(y[predictions == cluster]) for cluster in unique_clusters]
    sentiment_mapping = {
        cluster: 'Negative' if mean <= -0.33 else 'Positive' if mean >= 0.33 else 'Neutral'
        for cluster, mean in zip(unique_clusters, cluster_means)
    }
    
    sentiment_predictions = np.array([sentiment_mapping[label] for label in predictions])

    print("\nSample predictions:")
    for i in range(min(10, len(X))):
        print(f"Sample {i}: Predicted Cluster: {predictions[i]}, Predicted Sentiment: {sentiment_predictions[i]}, True Sentiment: {y[i]}")


    true_sentiments = np.array(['Negative' if s <= -0.33 else 'Positive' if s >= 0.33 else 'Neutral' for s in y])
    accuracy = np.mean(sentiment_predictions == true_sentiments)
    print(f"\nAccuracy: {accuracy:.2f}")


    np.savez('kmeans_results.npz', X=X, true_sentiment=y, predicted_cluster=predictions, predicted_sentiment=sentiment_predictions)

if __name__ == "__main__":
    main()