import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ------------------ Naive Bayes Classifier ------------------
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.std = {}
        self.prior = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean()
            self.std[c] = X_c.std().fillna(1e-6).replace(0, 1e-6)  # Avoid 0 or NaN std
            self.prior[c] = len(X_c) / len(X)

    def _gaussian(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            probs = {}
            for c in self.classes:
                prob = self.prior[c]
                for feature in X.columns:
                    prob *= self._gaussian(row[feature], self.mean[c][feature], self.std[c][feature])
                probs[c] = prob
            predictions.append(max(probs, key=probs.get))
        return predictions

# ------------------ Load & Preprocess Titanic Data ------------------
df = pd.read_csv(r'C:\Users\Gowtham M N\Downloads\titanic.csv')[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Fill missing values
# Fill missing values safely (no chained assignment)
df.fillna({
    'Age': df['Age'].median(),
    'Fare': df['Fare'].median(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)


# Convert 'Embarked' to numeric
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# ------------------ Run the Experiment ------------------
def run_experiment(test_size, label):
    print(f"\n--- {label} Split (Test size = {test_size}) ---")
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    X_train, y_train = train.drop('Survived', axis=1), train['Survived']
    X_test, y_test = test.drop('Survived', axis=1), test['Survived']

    model = NaiveBayes()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    acc = np.mean(predictions == y_test)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", round(acc, 4))

# ------------------ Try Two Different Splits ------------------
run_experiment(0.1, "90-10")
run_experiment(0.3, "70-30")