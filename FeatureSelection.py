from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a sample classification dataset
X, y = make_classification(n_samples=1000, n_features=32, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Select features based on importance scores
feature_selector = SelectFromModel(clf, threshold='median')
feature_selector.fit(X_train, y_train)

# Transform the training and testing data
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

# Print the selected feature indices
selected_indices = feature_selector.get_support(indices=True)
print("Selected Feature Indices:", selected_indices)

# Train and evaluate the model using the selected features
clf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
clf_selected.fit(X_train_selected, y_train)

accuracy = clf_selected.score(X_test_selected, y_test)
print("Accuracy with selected features:", accuracy)
