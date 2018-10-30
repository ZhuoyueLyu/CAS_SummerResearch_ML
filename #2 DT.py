# k-Nearest Neighbor （use model）
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# training data （除了那三个数据以外的数据）
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# testing data (仅用上面三个数据）
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 这行是test 集的真实label
print(test_target)
# 这行是model 基于数据，给予对于test 集label 的判断
print(clf.predict(test_data))

# You can print the data as following
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[1])
# print(iris.target[1])
#
# for i in range(0, 150):
#     print("Number:", i, "Data: ", iris.data[i], "Label: ", iris.target[i], "\n")

