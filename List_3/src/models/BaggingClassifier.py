import numpy as np
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier:
  def __init__(self, n_estimators=100, max_depth=None):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.trees = []
    
  def _sample(self, X, y):
    subset_indexes = np.random.choice(len(X), size=len(X), replace=True) # replace =true - zwracanie
    X_sub = X[subset_indexes]
    y_sub = y[subset_indexes]
      
    return (X_sub, y_sub)
    
  def fit(self, X, y):
    X_np = np.array(X)
    y_np = np.array(y)
    
    self.trees = [] # czyszczenie drzewa przy nwoym trenignu
    
    for _ in range(self.n_estimators):
      (X_sub, y_sub) = self._sample(X_np, y_np)
      
      tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=None)
      tree.fit(X_sub, y_sub)
      
      self.trees.append(tree)
    
  def _agregate(self, predictions, X_len):
    final_predictions = []
    
    for i in range(X_len):
      votes = predictions[:, i] # wszystkie wiersze z i-tej kolumny
      most_freq_vote = np.bincount(votes).argmax()
      final_predictions.append(most_freq_vote)
      
    return np.array(final_predictions)  
    
  def predict(self, X):
    X_np = np.array(X)
    
    predictions = np.array([tree.predict(X_np) for tree in self.trees])
    
    return self._agregate(predictions, X_np.shape[0])