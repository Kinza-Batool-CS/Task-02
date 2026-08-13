
# 1. Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# 2. Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target


# 3. Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# 4. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)


# 6. Train the model
knn.fit(X_train, y_train)


# 7. Make predictions
y_pred = knn.predict(X_test)


# 8. Check Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# 9. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 10. F1 Score
f1 = f1_score(y_test, y_pred, average="weighted")
print("\nF1 Score:", f1)