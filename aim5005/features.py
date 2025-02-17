import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        # Bug removed: (x- x.minimum)/ diff_max_min
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def _init_(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. 
        If it can't be cast, raise an error.
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array"
        return x

    def fit(self, x: np.ndarray) -> None:
        """
        Compute the mean and standard deviation for scaling.
        """
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x: np.ndarray) -> list:
        """
        Standardize the given vector.
        """
        x = self._check_is_array(x)

        # Prevent division by zero by replacing zero std with 1
        adjusted_std = np.where(self.std == 0, 1, self.std)

        return (x - self.mean) / adjusted_std

    def fit_transform(self, x: list) -> np.ndarray:
        """
        Compute the mean and std, then standardize the input.
        """
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """
        Fit label encoder, storing unique class labels.
        
        Parameters:
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        self : returns an instance of the fitted LabelEncoder.
        """
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """
        Transform labels to normalized encoding (integers).
        
        Parameters:
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns:
        y_transformed : array of shape (n_samples,)
            Integer representation of the labels.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder instance is not fitted yet. Call 'fit' before using this method.")
        
        y_transformed = np.array([np.where(self.classes_ == label)[0][0] for label in y])
        return y_transformed

    def fit_transform(self, y):
        """
        Fit label encoder and return encoded labels.
        
        Parameters:
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns:
        y_transformed : array of shape (n_samples,)
            Integer representation of the labels.
        """
        return self.fit(y).transform(y)
