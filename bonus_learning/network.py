import torch 



class Phi(torch.nn.Module):

    def __init__(self, state_n, action_n, repr=32):

        super().__init__()

        self.input_features = state_n+action_n
        self.output_features = repr
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.output_features)
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)

    
    def forward(self, x):

        out = self.net(x)
        return out 

class Psi(torch.nn.Module):

    def __init__(self, state_n, repr=32):

        super().__init__()

        self.input_features = state_n
        self.output_featuures = repr
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.input_features, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, self.output_features)
        )

        self.optimizer = torch.optim.Adam(self.net.paramerers(), lr = 0.001)

    def forward(self, x):

        out = self.net(x)
        return out
