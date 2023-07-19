from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a sample classification dataset
X, y = make_classification(n_samples=1000, n_features=32, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression classifier
classifier = LogisticRegression()

# Perform forward feature selection
selector = SequentialFeatureSelector(classifier, direction='forward')
selector.fit(X_train, y_train)

# Get selected feature indices
selected_indices = selector.get_support(indices=True)
print("Selected Feature Indices:", selected_indices)

# Transform the training and testing data to selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train the classifier on the selected features
classifier.fit(X_train_selected, y_train)

# Evaluate the classifier on the testing data
accuracy = classifier.score(X_test_selected, y_test)
print("Accuracy with selected features:", accuracy)
