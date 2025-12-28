import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
loss_plot = []
epochs_plot = []
class LogisticRegression:
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
            _loss = self.gradient_descent(learning_rate=learning_rate,predictions=predictions,input=X,output=y)
            print("Total Loss:",_loss)
            
            
            epochs_completed += 1
            epochs_plot.append(epochs_completed)
            loss_plot.append(_loss)
       
    def predict(self,input):
                product_w = self.weights * input
                prediction = product_w.sum() + self.bias
                prediction = 1/(math.e**(-prediction) + 1)
                return prediction
    def gradient_descent(self,predictions,input,output,learning_rate):
            n = len(predictions)
            predictions = pd.Series(predictions)
            _loss = 0
            series_of_difference = predictions - output
            loss_of_1s = (np.log(predictions))*output
            loss_of_0s = (np.log(1-predictions))*(1-output)
            _loss = np.sum([loss_of_0s,loss_of_1s])
            _loss = _loss/-n
            for i in range(0,self.col_amount):
                series_of_derivatives = series_of_difference*input.iloc[:,i]
                total_gradient = (series_of_derivatives.sum())/n
                step = total_gradient*learning_rate
                new_weight = self.weights[i] - step
                self.weights[i] = new_weight
            total_gradient_bias = (series_of_difference.sum())/n
            step = total_gradient_bias*learning_rate
            self.bias = self.bias - step
            return _loss
    def save(self):
         print("To save please note the returned values, and when creating a new linear regression object plug these values in as parameters, also pass the weights as a numpy array and if you want fit with your own params you can set useNewParams to False")
         print("weights: ",self.weights,"bias",self.bias)
         return {"weights":self.weights,"bias":self.bias}
data = {
    "feature1": [0.1, 1.2, -0.5, 0.3, -1.5, 0.7],
    "feature2": [0.5, -0.3, 0.7, 1.0, -0.8, -0.2],
    "target":   [0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data=data)
X = df.iloc[:,0:2]
y = df.iloc[:, 2]
my_model = LogisticRegression()
my_model.fit(X=X, y=y, epochs=1000, learning_rate=60)
plt.plot(epochs_plot,loss_plot)
plt.show()