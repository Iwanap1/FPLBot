import torch.nn as nn

# class XPModel(nn.Module):
#     def __init__(self, input_size, hidden_size=64):
#         super(XPModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 32)
#         self.fc3 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    

class XPModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.1, use_batchnorm=True):
        super().__init__()
        self.hidden_is_int = isinstance(hidden_size, int)

        if self.hidden_is_int:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
        else:
            assert len(hidden_size) >= 1, "hidden_size list must be non-empty"
            layers = [input_size] + list(hidden_size)

            # ensure final projection to 1 via a small head if not already 1
            if layers[-1] != 1:
                layers = layers + [1]  # we'll replace last layer below

            mods = []
            for i in range(len(layers) - 1):
                in_f, out_f = layers[i], layers[i+1]
                linear = nn.Linear(in_f, out_f)
                if i < len(layers) - 2:  # hidden
                    block = [linear]
                    if use_batchnorm:
                        block.append(nn.BatchNorm1d(out_f))
                    block += [nn.ReLU(), nn.Dropout(dropout)]
                    mods += block
                else:
                    # final layer to scalar
                    mods.append(nn.Linear(in_f, 1))
            self.layers = nn.Sequential(*mods)

    def forward(self, x):
        if self.hidden_is_int:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        else:
            return self.layers(x)
