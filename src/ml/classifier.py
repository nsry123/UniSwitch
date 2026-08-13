import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class TapClassifier:
    def __init__(self, n_neighbors: int = 3, distance_threshold: float = 2.5):
        """
        An incremental K-NN classifier with a strict rejection threshold.
        - n_neighbors: How many nearby samples to check for consensus.
        - distance_threshold: The maximum allowed Euclidean distance. 
          If the closest known tap is further than this, it's classified as 'Unknown'.
        """
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        
        # Scikit-learn's K-NN model
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        self.scaler = StandardScaler()
        
        # In-memory database of registered sounds
        self.X_data = []  # List of feature vectors (the extracted MFCCs/Centroids)
        self.y_labels = [] # List of string labels (e.g., 'desk_tap', 'wall_knock')
        
        self.is_trained = False

    def register_tap(self, feature_vector: np.ndarray, label: str):
        """
        Adds a new tap to the database and re-fits the model.
        Because K-NN is a lazy learner, 'fitting' is virtually instantaneous.
        """
        self.X_data.append(feature_vector)
        self.y_labels.append(label)
        
        # We can only use K-NN if we have at least 'K' samples in our dataset
        if len(self.X_data) >= self.n_neighbors:
            # Convert lists to numpy arrays for sklearn
            X_raw = np.array(self.X_data)
            y_train = np.array(self.y_labels)

            self.scaler.fit(X_raw)
            X_train = self.scaler.transform(X_raw)
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            print(f"[*] Model updated. Total registered samples: {len(self.X_data)}")
        else:
            print(f"[*] Sample recorded. Need {self.n_neighbors - len(self.X_data)} more to train.")

    def predict(self, feature_vector: np.ndarray) -> str:
        """
        Predicts the label of a new tap, but returns 'Unknown' if it 
        doesn't closely match any registered sounds.
        """
        if not self.is_trained:
            return "Unknown"

        # Sklearn expects a 2D array for prediction: shape (1, n_features)
        # vector_2d = feature_vector.reshape(1, -1)

        vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))

        # 1. Find the distance to the single closest neighbor
        distances, indices = self.model.kneighbors(vector_scaled, n_neighbors=1)
        closest_distance = distances[0][0]

        # 2. The Threshold Gate
        if closest_distance > self.distance_threshold:
            print(f"[-] Ignored: Unregistered sound (Distance: {closest_distance:.2f})")
            return "Unknown"

        # 3. If it passes the threshold, let the model predict normally
        prediction = self.model.predict(vector_scaled)[0]
        print(f"[+] Match found: {prediction} (Distance: {closest_distance:.2f})")
        
        return prediction