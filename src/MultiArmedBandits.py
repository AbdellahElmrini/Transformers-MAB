import numpy as np
from abc import ABC, abstractmethod

class MAB(ABC):
    def __init__(self):
        self.n = None

    @abstractmethod
    def pull(self, a):
        ...

class MAB_normal(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N(a,1)
        """
        super().__init__()
        self.n = n
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        return np.random.rand()+ a

        