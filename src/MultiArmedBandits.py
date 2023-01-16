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
        return np.random.randn()+ a

class MAB_normal2(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N((a+1)/(n+2),1/n)
        """
        super().__init__()
        self.n = n
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        return 1/np.sqrt(n)*np.random.randn() + (a+1)/(n+1)

class MAB_normal_random_order(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N((a+1)/(n+2),1/n)
        """
        super().__init__()
        self.n = n
        self.perm = np.arange(n)
        np.random.shuffle(self.perm)
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        return 1/np.sqrt(n)*np.random.randn() + (self.perm[a]+1)/(n+1)
        
class MAB_Bernoulli(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form B((a+1)/(n+2))
        """
        super().__init__()
        self.n = n

    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        return np.random.rand() < (a+1)/(n+2)
