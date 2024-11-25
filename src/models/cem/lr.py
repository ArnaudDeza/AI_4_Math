import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, num_edges):
        """
        Logistic Regression model for sequential graph construction.

        Args:
            args: An object containing model arguments.
                args.model_args: Dictionary with architecture-related parameters:
                    - None required for logistic regression.
        """
        super(LogisticRegressionModel, self).__init__()
        self.E = num_edges  # Number of edges E

        # Linear layer without bias
        self.linear = nn.Linear(2 * self.E, 1, bias=True)

    def forward(self, x):
        """
        Forward pass of the Logistic Regression model.

        Args:
            x: Input tensor of shape (batch_size, 2 * E).

        Returns:
            logit: Tensor of shape (batch_size, 1) representing the logit for the binary decision.
        """
        logit = self.linear(x)  # (batch_size, 1)
        # apply sigmoid
        logit = torch.sigmoid(logit)
        return logit