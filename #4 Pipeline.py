from sklearn import datasets
iris = datasets.load_iris()


X = iris.data
y = iris.target

# 下面这行是把训练集分成训练+test两个集的工具
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# 下面就是classifier了(可以有多种）
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
#  用KNN
from sklearn.neighbors import KNeighborsClassifier
my_classifier2 = KNeighborsClassifier()


# training
my_classifier.fit(X_train, y_train)
my_classifier2.fit(X_train, y_train)
# predit
predictions = my_classifier.predict(X_test)
predictions2 = my_classifier2.predict(X_test)
# 计算正确率（也是直接调用了一个method）
from sklearn.metrics import accuracy_score
print(accuracy_score(predictions, y_test))
print(accuracy_score(predictions2, y_test))

