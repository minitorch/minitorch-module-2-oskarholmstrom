"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # Apply first linear layer followed by ReLU activation
        h1 = self.layer1.forward(x).relu()
        
        # Apply second linear layer followed by ReLU activation
        h2 = self.layer2.forward(h1).relu()
        
        # Apply final linear layer followed by sigmoid activation
        return self.layer3.forward(h2).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # Extract dimensions from input tensor
        batch_size, in_features = x.shape
        
        # Reshape weights to allow broadcasting across the batch dimension
        # Shape goes from (in_features, out_features) to (1, in_features, out_features)
        reshaped_weights = self.weights.value.view(1, in_features, self.out_size)
        
        # Reshape input to prepare for element-wise multiplication with weights
        # Shape goes from (batch_size, in_features) to (batch_size, in_features, 1)
        reshaped_input = x.view(batch_size, in_features, 1)
        
        # Perform element-wise multiplication and sum along the input feature dimension (dim=1)
        # This effectively computes the dot product for each sample and output feature
        weighted_sum = (reshaped_weights * reshaped_input).sum(1)
        
        # Reshape the result to the expected output shape (batch_size, out_features)
        output = weighted_sum.view(batch_size, self.out_size)
        
        # Reshape bias for broadcasting and add to the output
        reshaped_bias = self.bias.value.view(self.out_size)
        
        # Return the final result with bias added
        return output + reshaped_bias


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
