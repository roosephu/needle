
class BasicAgent:
    def __init__(self, input_dim, output_dim):
        pass

    def init(self):
        pass

    def reset(self, save=False):
        pass

    def action(self, state, show=False):
        '''
            Input:
                @state: of shape (1, input_dim)
            Output:
                @action: of shape (1, )
        '''
        pass

    def feedback(self, state, action, reward, done, new_state):
        '''
            Input:
                @state: of shape (1, input_dim)
                @action: of shape (1, )
                @reward: integer
                @done: boolean
                @new_state: of shape (1, input_dim)
        '''
        pass

    def train(self):
        pass