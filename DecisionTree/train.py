import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Node:
    def __init__(self,feature,unique_value,leaf, depth=1):
        self.feature = feature
        self.unique_value = unique_value
        self.depth = depth
        self.leaf = leaf
    def set_right_node(self,right_node):
        if self.leaf:
            return
        right_node.depth = self.depth+1
        self.right_node = right_node
    def set_left_node(self,left_node):
        left_node.depth = self.depth+1
        self.left_node = left_node
class DecisionTree:
    def __init__(self, tree_depth=3):
        self.tree_depth = tree_depth
    def fit(self,train:pd.DataFrame,y:int):
        X = train.drop(df.columns[y], axis=1)
        y_train = train.iloc[:,y]
        lowest_entropy = None
        for i in range(0,X.shape[1]):
            for value in X.iloc[:,i].unique():
                rows = train[train[train.columns[i]] == value]
                print(rows)
df = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Rainy", "Rainy"],
    "Temp": ["Hot", "Mild", "Mild", "Cold"],
    "Play": ["No", "Yes", "Yes", "No"]
})
label = 2
my_model = DecisionTree()
my_model.fit(train=df,y=label)