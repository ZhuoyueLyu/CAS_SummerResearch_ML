from __future__ import print_function

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]
header = ["color", "diameter", "label"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset.
    即，把给的数据集，每个sample的第col个元素都提取出来(即，看看有哪些，要求是不重合的）
    >>> unique_vals(training_data, 0)
    {'Yellow', 'Green', 'Red'}
    >>> unique_vals(training_data, 1)
    {3, 1}
    """
    return set(x[col] for x in rows)


def class_counts(rows):
    """Counts the number of each type of example in a dataset.
    即，每种水果出现了几次(这里用if label not in counts 很妙）
    >>> class_counts(training_data)
    {'Apple': 2, 'Grape': 2, 'Lemon': 1}
    """
    counts = {}
    for i in rows:
        label = i[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    即第一个数字是代表哪个column，比如0，就是颜色的column，而后一项就是具体的值，比如“绿色”，然后其实具体的工作是“match”在做(需要叫q.match才出来)而__repr__只是负责在呈现这样一个str的问句（叫q即可出来）
    >>> q = Question(0, 'Green')
    >>> q
    Is color == Green?
    >>> example = training_data[0]
    >>> q.match(example)
    True
    >>> Question(1, 3)
    Is diameter >= 3?
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        """
        # Compare the feature value in an example to the
        # feature value in this question.
        """
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


def partition(rows, question):
    """
    即，把dataset，根据就question而言的正误分开来
    Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.

    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """
    quantify how much uncertainty(chance of incorrect, impurity) there is at a node
    (impurity:
    Chance of being incorrect if you randomly assign a label to an example in the same set)
    按道理来说，你正确的label是啥，是应该会影响其chance率的，但下面这个例子，其实就不管你是什么例子，就计算出一个impurity来了
    """
    counts = class_counts(rows)
    # 即，counts是一个dic,比如{'Apple': 2, 'Grape': 2, 'Lemon': 1}
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        # counts[lbl]直接导出对应的数字了，因为lbl是字典的index
        impurity -= prob_of_lbl ** 2
        # 即，对于每一项, impurity要减去其label正确的
    return impurity


def info_gain(left, right, current_uncertainty):
    """
    let us quantify how much a question reduces that
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    >>>current_uncertainty = gini(training_data)
    >>>current_uncertainty
    0.6399999999999999
    #How much information do we gain by partioning on 'Green'?
    >>>true_rows, false_rows = partition(training_data, Question(0, 'Green'))
    >>>info_gain(true_rows, false_rows, current_uncertainty)
    0.1399999999999999
    # What about if we partioned on 'Red' instead?
    >>>true_rows, false_rows = partition(training_data, Question(0,'Red'))
    >>>info_gain(true_rows, false_rows, current_uncertainty)
    0.37333333333333324
    #所以这里看来是问 “Red” 比问 “Green” 能gain更多information，并且自己分下去也是这样的，问Green,有一侧只有一个元素了。
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    即：要问出最棒的问题，需要遍历所有的特征、数值，并且计算出其Information gain
    >>>best_gain, best_question = find_best_split(training_data)
    >>>best_question
    Is diameter >= 3?
    """
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep track of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns, 即除了label之外的column有几个（所以-1），也就是features有几个

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])
        # 即，在上面for的每个feature下,把每一个example的这一个feature的值放在value这个set下。
        for val in values:  # for each value 对于每一个example的这一个feature的值中的每一个。
            question = Question(col, val)  # for each feature 都提问一遍
            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question
            # 然后持续根据最佳的信息增益来更新问题。
        return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """
    Built the tree
    其实就是利用到上学期学的recursion了
    """
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


#######
# Demo:
# On the second example, the confidence is lower
# print_leaf(classify(training_data[1], my_tree))
#######

if __name__ == '__main__':

    my_tree = build_tree(training_data)

    print_tree(my_tree)

    # Evaluate
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    for row in testing_data:
        print("Actual: %s. Predicted: %s" %
              (row[-1], print_leaf(classify(row, my_tree))))

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for regression
