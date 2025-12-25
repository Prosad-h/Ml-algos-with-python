import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Node:
    def __init__(self,feature,unique_value,leaf_value=None, depth=1):
        self.feature = feature
        self.unique_value = unique_value
        self.depth = depth
        self.leaf = leaf_value
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
        self.root_node = self.split_node(train,y)
    def split_node(self,data:pd.DataFrame,label_index:int)-> Node:
        if len(data.iloc[:,label_index].unique()) == 1:
            return Node(feature=None,unique_value=None,leaf_value=data.iloc[:,label_index].unique()[0])
        X = data.drop(data.columns[label_index],axis=1)
        lowest_entropy = 2
        required_feature = None
        required_value = None
        for i in range(0,X.shape[1]):
            for value in X.iloc[:,i]:
                split_df = data[data[data.columns[i]] == value]
                probs = []
                counts = split_df.iloc[:,label_index].value_counts()
                for n in counts:
                    prob = n/split_df.shape[0]
                    prob = prob*np.log2(prob)
                    probs.append(prob)
                probs = pd.Series(probs)
                entropy = -probs.sum()
                if entropy < lowest_entropy:
                    lowest_entropy = entropy
                    required_feature = i
                    required_value = value
        final_node = Node(feature=required_feature,unique_value=required_value)
        left_df = data[data[data.columns[required_feature]] == required_value]
        right_df = data[data[data.columns[required_feature]] != required_value]
        left_node = self.split_node(data=left_df,label_index=label_index)
        right_node = self.split_node(data=right_df,label_index=label_index)
        final_node.set_left_node(left_node=left_node)
        final_node.set_right_node(right_node=right_node)
        return final_node
    def predict(self,x:list):
        current_node = self.root_node
        while True:
            if current_node.leaf != None:
                return current_node.leaf
            if x[current_node.feature] == current_node.unique_value:
                current_node = current_node.left_node
            else:
                current_node = current_node.right_node
                
                

df = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Rainy", "Rainy"],
    "Temp": ["Hot", "Mild", "Mild", "Cold"],
    "Play": ["No", "Yes", "Yes", "No"]
})

label = 2

my_model = DecisionTree()
my_model.fit(train=df,y=label)
print(my_model.predict(["Sunny","Mild"]))