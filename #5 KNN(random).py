# # k-Nearest Neighbor （by myself）
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


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
        :param x_train:
        :type x_train:
        :param y_train:
        :type y_train:
        :return:
        :rtype:
        """

    def predict(self, x_test):
        """
        :param x_test:
        :type x_test:
        :return:
        :rtype:
        a function that predict a never-before-seen data x_test
        """
        predictions = []
        for row in x_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions


my_classifier = ScrappyKNN()
# training
my_classifier.fit(x_train, y_train)

# predit
predictions = my_classifier.predict(x_test)

# 计算正确率（也是直接调用了一个method）
print(accuracy_score(predictions, y_test))
