
import numpy as np
import pandas as pd
import math

from requests import head



class LinearRegressionModel:
    def __init__(self,weights=[],bias=0):
        self.weights = weights
        self.bias = bias
    def fit(self,X:pd.DataFrame,y:pd.Series,learning_rate=0.001, epochs=100,useNewParams=True):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a data frame")
        self.col_amount = X.shape[1]
        if useNewParams:
            self.weights = np.random.rand(self.col_amount)
            self.bias = np.random.rand(1)[0]
        epochs_completed = 0
        while epochs_completed != epochs:
            predictions = []
            for i in range(0,X.shape[0]):
                final_prediction = self.predict(X.iloc[i])
                predictions.append(final_prediction)
            print("Total Loss:",(self.gradient_descent(learning_rate=learning_rate,predictions=predictions,input=X,output=y))**0.5)
            
            epochs_completed += 1
       
    def predict(self,input):
                product_w = self.weights * input
                return product_w.sum() + self.bias
    def gradient_descent(self,predictions,input,output,learning_rate):
         n = len(predictions)
         
         _loss = 0
         for i in range(0,self.col_amount):
            series_of_loss = predictions - output
            _loss = series_of_loss**2
            
            _loss = _loss.sum()
            _loss = _loss/n
           
            series_of_derivatives = series_of_loss*input.iloc[:,i]
            total_loss = (series_of_derivatives.sum()*2)/n
            step = total_loss*learning_rate
            new_weight = self.weights[i] - step
            self.weights[i] = new_weight
         return _loss
    def save(self):
         print("To save please note the returned values, and when creating a new linear regression object plug these values in as parameters, also pass the weights as a numpy array and if you want fit with your own params you can set useNewParams to False")
         print("weights: ",self.weights,"bias",self.bias)
         return {"weights":self.weights,"bias":self.bias}
        
df = pd.read_csv("/winequality-red.csv")
df_train = df.iloc[0:math.floor(df.shape[0]*0.8)]
df_test = df.iloc[math.floor(df.shape[0]*0.8):-1]
X_train = df_train.iloc[:,0:-1]
y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,0:-1]
y_test = df_test.iloc[:,-1]
my_model = LinearRegressionModel()
my_model.fit(y=y_train,X=X_train, learning_rate=0.00001)
prediction = my_model.predict(X_test.iloc[4])
print("Prediction:",prediction,"Actual Value:",y_test.iloc[4])
my_model.save()