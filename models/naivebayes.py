import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = BernoulliNB()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.labels = None

    def load_data(self, X, Y):
        self.features = np.load(X, allow_pickle=True)
        self.labels = np.load(Y, allow_pickle=True)

    def train(self):
        print("Shape of features:", np.shape(self.features))
        print("Shape of labels:", np.shape(self.labels['rounded']))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.labels['rounded'], test_size=0.2, random_state=42)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.X_test)

    def evaluate(self):
        y_pred = self.classifier.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)
        
        return accuracy, report

    def visualize(self):
        X_test_vectorized = self.vectorizer.transform(self.X_test)
        y_pred = self.classifier.predict(X_test_vectorized)
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


        # all_words = [word for text in self.X_train for word in text.split()]
        # word_counts = Counter(all_words)

        # # Get top N most common words as features
        # N = 1000  # for example
        # top_words = [word for word, count in word_counts.most_common(N)]
        # # Get feature importance
        # feature_importance = self.classifier.feature_log_prob_[1] - self.classifier.feature_log_prob_[0]

        # # Sort features by importance
        # sorted_idx = np.argsort(feature_importance)
        # top_features = np.array(top_words)[sorted_idx[-10:]]
        # top_importance = feature_importance[sorted_idx[-10:]]

        # plt.figure(figsize=(10,7))
        # plt.barh(top_features, top_importance)
        # plt.title('Top 10 Most Important Features')
        # plt.xlabel('Feature Importance')
        # plt.ylabel('Features')
        # plt.show()

if __name__ == "__main__":
    
    analyzer = SentimentAnalyzer()
    analyzer.load_data('X_1000.npy', 'y.npz')
    analyzer.train()
    analyzer.evaluate()
    analyzer.visualize()
   