import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Name : Omar Abdullah Saeed 
# ID : 20210706 

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Make the function split by my hand :
def train_validate_test_split(data, labels, testRatio=0.3, valRatio=0.3):
    assert testRatio + valRatio < 1, "testRatio + valRatio should be less than 1"
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=(testRatio + valRatio), random_state=42)
    valSplit = valRatio / (testRatio + valRatio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=valSplit, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(X, y)


gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_val_pred = gnb.predict(X_val)
y_test_pred = gnb.predict(X_test)

# Make the function accuracy by my hand :
def calculate_accuracy(predicted_y, y):
    assert len(predicted_y) == len(y), "The lengths of predicted_y and y should be the same."
    correct = np.sum(predicted_y == y)
    accuracy = correct / len(y)
    return accuracy

# Calculate accuracy for the validation and test sets
val_accuracy = calculate_accuracy(y_val_pred, y_val)
test_accuracy = calculate_accuracy(y_test_pred, y_test)

# Plot decision boundaries
X_2d = X_train[:, :2]
gnb.fit(X_2d, y_train)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, marker='o', s=25)
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("The Graph")


if __name__ == "__main__":
    print("Validation Accuracy:", "{:.3f}".format(val_accuracy))
    print("Test Accuracy:", "{:.3f}".format(test_accuracy))
    plt.show()
    