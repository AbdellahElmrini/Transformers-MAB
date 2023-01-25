import numpy as np
import torch
from abc import ABC, abstractmethod

#TODO Assert action is among possible actions
#TODO Pull multiple arms at the same time (for vectorization)



class MAB(ABC):
    def __init__(self):
        self.n = None
        self.best_action = None

    @abstractmethod
    def pull(self, a):
        ...
    
    @abstractmethod
    def pull_V(self, A):
        ...

class MAB_normal_V0(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N(a,1)
        """
        super().__init__()
        self.n = n
        self.best_action = n-1
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        return np.random.randn()+ a

    def pull_V(self, A):
        """
        A (torch.Tensor) : Vectorized version of pull
        """
        n = self.n
        m = A.size(0)
        return np.random.randn(m,1) + (A.numpy()+1)/(n+1)

class MAB_normal_V1(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N((a+1)/(n+2),1/n)
        """
        super().__init__()
        self.n = n
        self.best_action = n-1
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        return 1/np.sqrt(n)*np.random.randn() + (a+1)/(n+1)
    
    def pull_V(self, A):
        """
        A (torch.Tensor) : Vectorized version of pull
        """
        n = self.n
        m = A.size(0)
        return 1/np.sqrt(n)*np.random.randn(m,1) + (A.numpy()+1)/(n+1)

class MAB_normal(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form N((a+1)/(n+2),1/n)
        Order of the arms is arbitrary this time
        """
        super().__init__()
        self.n = n
        self.perm = np.arange(n)
        np.random.shuffle(self.perm)
        self.best_action = list(self.perm).index(n-1)
        self.best_reward_avg = (n+1)/(n+2)
    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        return 1/np.sqrt(n)*np.random.randn() + (self.perm[a]+1)/(n+1)
    
    def pull_V(self, A):
        """
        A (torch.Tensor) : Vectorized version of pull
        """
        n = self.n
        #assert A>= 0 and A <= n-1
        m = A.size(0)
        return 1/np.sqrt(n)*np.random.randn(m,1) + (self.perm[A]+1)/(n+1)
        
class MAB_Bernoulli_V0(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form B((a+1)/(n+2))
        """
        super().__init__()
        self.n = n
        self.best_action = n-1
        self.best_reward_avg = (n+1)/(n+2)

    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        return np.random.rand() < (a+1)/(n+2)
    
    def pull_V(self, A):
        """
        A (torch.Tensor) : Vectorized version of pull
        """
        n = self.n
        #assert A>= 0 and A <= n-1
        m = A.size(0)
        return np.random.rand(m, 1) < (A.numpy()+1)/(n+2)

class MAB_Bernoulli(MAB):
    def __init__(self, n):
        """
        A special multi-armed bandits, with n arms giving awards of the form B((a+1)/(n+2))
        """
        super().__init__()
        self.n = n
        self.perm = np.arange(n)
        np.random.shuffle(self.perm)
        self.best_action = list(self.perm).index(n-1)
        self.best_reward_avg = (n+1)/(n+2)

    def pull(self, a):
        """
        a (int): Chosen arm in 0,..,n-1
        """
        n = self.n
        assert a>= 0 and a<= n-1, "Action not possible"
        return np.random.rand() < (self.perm[a]+1)/(n+2)

    def pull_V(self, A):
        """
        A (torch.Tensor) : Vectorized version of pull
        """
        n = self.n
        #assert A>= 0 and A <= n-1
        m = A.size(0)
        return np.random.rand(m, 1) < (self.perm[A]+1)/(n+2)
