# Module file for implementation of ID3 algorithm.

# You can add optional keyword parameters to anything, but the original
# interface must work with the original test file.
# You will of course remove the "pass".

import os, sys
import numpy
from collections import Counter
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle
# You can add any other imports you need.

class DecisionTree:
    def __init__(self, load_from=None):
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.decision_tree = pickle.load(load_from)

    # calculate target class entropy
    def target_entropy(self, target):
        entropy_list = []
        for i in Counter(target).values():
            entropy_list.append(i / math.fsum(Counter(target).values()) * math.log2(i / math.fsum(Counter(target).values())))
        target_entropy = - math.fsum(entropy_list)
        return target_entropy

    # calculate every feature class entropy
    def entropy(self, feature, target):
        counts_list = []
        entropy_list = []
        label_counts_dict = dict(Counter(list(zip(feature, target))))
        counts_dict = dict(Counter(feature))
        for label, count in label_counts_dict.items():
            counts_list.append({label[0]: (-(label_counts_dict[label] / counts_dict[label[0]] *
                                             math.log2(label_counts_dict[label] / counts_dict[label[0]])))})
        new_counts_list = Counter()
        for item in counts_list:
            for key, value in item.items():
                new_counts_list[key] += value
        for key, value in new_counts_list.items():
            entropy_list.append((value * counts_dict[key] / sum(counts_dict.values())))
        return sum(entropy_list)

    # find max information gain
    def find_max_gain(self, data, class_column):
        gains = {}
        for column_name in data:
            gains[column_name] = (self.target_entropy(class_column) - self.entropy(data[column_name], class_column))
        return max(gains, key=gains.get)


    def train(self, X, y, attrs, prune=False):
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.
        # *********************************************************************
        #
        # I've referred to this article for some clarification:
        # https://www.python-course.eu/Decision_Trees.php.
        # Specifically, how they represent the model in a dictionary.

        label_list = []
        for label in y:
            label_list.append(label)
        # check if all examples in the data have the same label -> return that label
        if len(set(label_list)) <= 1:
            return label
        # check number of attributes
        elif len(attrs) == 0:
            return max(set(label_list), key=label_list.count)
        # otherwise begin
        else:
            # find feature with max gain
            max_gain = self.find_max_gain(X, y)
            decision_tree = {max_gain: {}}
            if max_gain in attrs:
                attrs.remove(max_gain)
            for value in set(X[max_gain]):
                tree_branch = self.train(X.loc[X[max_gain] == value], (y[X[max_gain] == value]), attrs)
                decision_tree[max_gain][value] = tree_branch
            self.decision_tree = decision_tree
            self.most_common_value = y.mode()
            return self.decision_tree

    def predict(self, instance, tree, most_common_value):
        # Returns the class of a given instance.
        #  Raise a ValueError if the class is not trained.
        if type(tree) == dict:
            try:
                for key, value in tree.items():
                    root_node = key
                    if type(value) == dict:
                        root_node_value = instance[root_node]
                        return self.predict(instance, tree[root_node][root_node_value], most_common_value)
                    else:
                        result = value

            except KeyError:
                result = most_common_value

        else:
            result = tree
        return result

    def test(self, X, y, display=False):
        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.
        instances = []
        results = []
        test_result = X.copy()
        for index, row in test_result.iterrows():
            instances.append(row)
            results.append(self.predict(row, self.decision_tree, self.most_common_value))
        test_result["result"] = [x for x in results]
        precision_recall_fscore = (precision_recall_fscore_support(y, test_result["result"], average="weighted"))
        ev_result = {'precision':precision_recall_fscore[0],
                  'recall':precision_recall_fscore[1],
                  'accuracy':accuracy_score(y, test_result["result"]),
                  'F1':precision_recall_fscore[2],
                  'confusion-matrix':confusion_matrix(y, test_result["result"])}
        if display:
            print(ev_result)
        return ev_result

    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        if self.decision_tree == None:
            return "ID3 untrained"
        else:
            return str(self.decision_tree)

    def save(self, output):
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pickle_model = pickle.dumps(self.decision_tree)
        output.write(pickle_model)
        output.close()
