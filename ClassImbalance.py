from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate a sample imbalanced classification dataset
X, y = make_classification(n_samples=1000, n_features=32, weights=[0.95, 0.05], random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Original class distribution
print("Original class distribution:")
print("Class 0:", sum(y_train == 0))
print("Class 1:", sum(y_train == 1))

# Train a Random Forest classifier on the original imbalanced data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
original_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on original imbalanced data:", original_accuracy)

# Random Under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
clf_rus = RandomForestClassifier(random_state=42)
clf_rus.fit(X_resampled, y_resampled)
y_pred_rus = clf_rus.predict(X_test)
rus_accuracy = accuracy_score(y_test, y_pred_rus)
print("Accuracy after Random Under-sampling:", rus_accuracy)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_resampled, y_resampled)
y_pred_smote = clf_smote.predict(X_test)
smote_accuracy = accuracy_score(y_test, y_pred_smote)
print("Accuracy after SMOTE:", smote_accuracy)

# ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
clf_adasyn = RandomForestClassifier(random_state=42)
clf_adasyn.fit(X_resampled, y_resampled)
y_pred_adasyn = clf_adasyn.predict(X_test)
adasyn_accuracy = accuracy_score(y_test, y_pred_adasyn)
print("Accuracy after ADASYN:", adasyn_accuracy)

# Cluster Centroids
cc = ClusterCentroids(random_state=42)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
clf_cc = RandomForestClassifier(random_state=42)
clf_cc.fit(X_resampled, y_resampled)
y_pred_cc = clf_cc.predict(X_test)
cc_accuracy = accuracy_score(y_test, y_pred_cc)
print("Accuracy after Cluster Centroids:", cc_accuracy)

# Evaluate classification report for imbalanced data
print("\nClassification report on original imbalanced data:")
print(classification_report_imbalanced(y_test, y_pred))

# Evaluate classification report after SMOTE
print("Classification report after SMOTE:")
print(classification_report_imbalanced(y_test, y_pred_smote))
