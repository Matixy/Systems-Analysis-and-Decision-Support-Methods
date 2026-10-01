import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

class MoEModel:
  def __init__(self, n_experts=3):
    self.n_experts = n_experts
    self.router = RandomForestClassifier(max_depth=5, random_state=42)
    self.experts = {}
    
    self.kmeans = KMeans(n_clusters=self.n_experts, random_state=42, n_init=10)
  
  def fit(self, X, y):
    X_np = np.array(X)
    y_np = np.array(y)
    
    #klasteryzacja
    cluster_labels = self.kmeans.fit_predict(X_np)
    
    #trenowanie routera
    self.router.fit(X_np, cluster_labels)
      
    #trenowanie expertow
    for i in range(self.n_experts):
      #wyciagniecie danych nalezacych tylko do klastra i
      mask = (cluster_labels == i)
      X_cluster = X_np[mask]
      y_cluster = y_np[mask]
      
      expert = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
      
      # jezeli klaster jest pusty nie jest trenowany
      if len(X_cluster) > 0:
        expert.fit(X_cluster, y_cluster)
        self.experts[i] = expert # dodanie wytrenowanego experta do tablicy
  
  def predict(self, X):
    X_np = np.array(X)
    num_of_clients = len(X_np)
    
    predictions = np.zeros(num_of_clients)
    
    # router decyduje do jakiego eksperta wyslac danego klienta
    assigned_clusters = self.router.predict(X_np)
    
    for i in range(num_of_clients):
        # reshape(1, -1) potrzebne- model wymaga zawsze tabeli 2D
        client_data = X_np[i].reshape(1, -1) 
        expert = self.experts[assigned_clusters[i]]
        prediction = expert.predict(client_data)
        
        predictions[i] = prediction[0]
    
    return predictions
  
  def get_kmeans_model(self):
    return self.kmeans