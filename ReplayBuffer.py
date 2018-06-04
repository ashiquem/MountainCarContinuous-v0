import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed size buffer to store experience tuple"""

    def __init__(self,buffer_size,batch_size):
        """Initialize replay buffer.

        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch

        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state",
        "done"])

    def add_experience(self,state,action,reward,next_state,done):
        """Add a new experience to memory"""
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)

    def sample(self,batch_size=64):
        """Randomly sample experience from memory"""
        return random.sample(self.memory,k = self.batch_size)

    def __len__(self):
        """Return length of buffer"""
        return len(self.memory)