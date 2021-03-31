import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        #=== My changes ====
        self.epsilon_lb = 0.001
        self.epsilon = 0.1     # epsilon-greedy policy parameters (exploitation vs exploration) 
                                   # espsilon=1 yields e-greedy with equiprobable random policy
                                   # espsilon=0 yields the greedy policy that most favors exploitation over exploration
        self.alpha = 0.35      # update rate
        self.n_episodes = 1    # episodes counter initialization
        self.gamma = 0.97      # discount rate 
        # ==================

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p=self.e_greedy_policy(state))  #return np.random.choice(self.nA)
    
    # =====  My changes  =====
    
    def e_greedy_policy(self, state):
        """epsilon-greedy policy
        Given the state, find the action policy 
        """        
        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])] += 1-self.epsilon
        
        return policy
    # ========================

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # === My changes ====
        if done :
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.n_episodes += 1
            self.epsilon =  1. / self.n_episodes   #self.epsilon_lb +     # decreasing epsilon over episodes #self.epsilon_lb +
        else:
            self.Q[state][action] += self.alpha * (reward + self.gamma * (np.max(self.Q[next_state]-self.Q[state][action])))
        # ===================
