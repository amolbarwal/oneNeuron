import numpy as np

class Perceptron:
  def __init__(self, eta, epochs):
    np.random.seed(42)
    self.weights = np.random.randn(3) * 1e-4
    print(f'initial weights before training: {self.weights}')
    self.eta = eta #learning rate
    self.epochs = epochs
  
  def activation_function(self, inputs, weights):
    z = np.dot(inputs, weights) # z = W*x 
    return np.where(z>0,1,0)
  
  def fit(self, X, y):
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, np.ones((len(self.X),1))]
    print(f'X with bias: \n{X_with_bias}')

    for epoch in range(self.epochs):
      print('--'*10)
      print(f'for epoch: {epoch}')
      print('--'*10)

      y_hat = self.activation_function(X_with_bias, self.weights) #forward propagation

      print(f'Predicted value after forward pass: {y_hat}')

      self.error = self.y - y_hat
      print(f'error: \n{self.error}')

      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)  #Backward Propagation
      print(f'Updated weights after epoch: {epoch}/{self.epochs} : {self.weights}')
      print('#####'*10)


  def predict(self, X):
    X_with_bias = np.c_[X, np.ones((len(X),1))]
    return self.activation_function(X_with_bias, self.weights)
  
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"Total loss: {total_loss}")
    return total_loss