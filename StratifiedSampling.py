from sklearn.model_selection import train_test_split

# Assuming you have your features in X and labels in y

# Split the data into training and testing sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Check the class distribution in the original data
class_distribution_original = y.value_counts(normalize=True)
print("Original Class Distribution:")
print(class_distribution_original)

# Check the class distribution in the training data after stratified sampling
class_distribution_train = y_train.value_counts(normalize=True)
print("\nClass Distribution in Training Data after Stratified Sampling:")
print(class_distribution_train)

# Check the class distribution in the testing data after stratified sampling
class_distribution_test = y_test.value_counts(normalize=True)
print("\nClass Distribution in Testing Data after Stratified Sampling:")
print(class_distribution_test)
