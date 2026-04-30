import torch 
import gymnasium as gym 

torch.set_default_dtype(torch.float64)


class Actor(torch.nn.Module):

    def __init__(self, state_n, action_n, action_space):

        super().__init__()
        self.input_features = state_n 
        self.output_features = action_n 
        self.action_space = action_space 


        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 400), 
            torch.nn.ReLU(), 
            torch.nn.Linear(400, 300), 
            torch.nn.ReLU(), 
            torch.nn.Linear(300, self.output_features)
        )


        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.0003)

    
    def forward(self, state):

        out = self.net(state)
        error = torch.normal(mean = 0, std = 0.1, size = out.shape).to('cuda')
        out = out + error 
        out = torch.tanh(out)
        return out 

    def forward_pred(self, state):

        out = self.net(state)
        error = torch.normal(mean = 0, std = 0.2, size = out.shape).to('cuda')
        error = torch.clamp(error, min = -0.5, max = 0.5)
        out = out + error 
        out = torch.tanh(out)
        return out 

class QFunction(torch.nn.Module):

    def __init__(self, state_n_action_n):

        super().__init__()
        self.input_features = state_n_action_n 

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 400), 
            torch.nn.ReLU(), 
            torch.nn.Linear(400, 300), 
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1)
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.0003)

    def forward(self, input_x):

        out = self.net(input_x)
        return out