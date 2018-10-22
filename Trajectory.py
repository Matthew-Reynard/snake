class Traj():
    
    def __init__(self, state, action, reward, new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
