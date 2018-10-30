# k-Nearest Neighbor （by myself） k = 1
# Pros: simple
# Cons:
# computationally intensive(have to iterate through all the points)
# some features are more informative than others, but it's not easy to represent that in KNN.

from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


def euc(a, b):
    """
    Calculate the Euclidean distance between points a and b
    """
    return distance.euclidean(a, b)


class ScrappyKNN:
    """
    A scrappy k-Nearest Neighbor classifier
    """
    def __init__(self):
        """
        Define what's in this class
        """
        self.x_train = x_train
        self.y_train = y_train

    def fit(self, x_train, y_train):
        """
        a function that train the model
        """

    def predict(self, x_test):
        """
        a function that predict a never-before-seen data x_test
        """
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        """
        find the closest point to the test point
        """
        label = y_train[0]
        shortest_dis = euc(row, self.x_train[0])
        for i in range(len(x_train)):
            if euc(row, self.x_train[i]) <= shortest_dis:
                shortest_dis = euc(row, self.x_train[i])
                label = y_train[i]
        return label


my_classifier = ScrappyKNN()
# training
my_classifier.fit(x_train, y_train)

# predit
predictions = my_classifier.predict(x_test)

# 计算正确率（直接调用了一个method）
print(accuracy_score(predictions, y_test))
print("成功了，哈哈")
