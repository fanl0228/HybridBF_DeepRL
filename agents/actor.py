import torch
import torch.nn as nn
import torch.distributions

from agents.backboneRL import BackboneRL


class Actor(nn.Module):
    """ actor network."""

    def __init__(
        self, 
        state_shape,    # [batch, Doppler, frame, range, ant]
        action_shape,     # [tx/rxbf, beams] = [bs, 2, 121]
        dropout_rate=None,
        log_std_min=-10.0, 
        log_std_max=2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.nDoppler = state_shape[1]
        self.txbf_output_dim = action_shape[1]
        self.rxbf_output_dim = action_shape[2]
        
        self.backbone = BackboneRL(model_depth=50, nDoppler=self.nDoppler, output_dim=2048)

        self.fc_txbf = nn.Linear(2048, self.txbf_output_dim)
        self.fc_rxbf = nn.Linear(2048, self.rxbf_output_dim)

        self.softmax_txbf = torch.nn.Softmax(dim=1)
        self.softmax_rxbf = torch.nn.Softmax(dim=1)
        
    def forward(self, state):
        x = self.backbone(state)
        
        x_txbf = self.fc_txbf(x)
        x_rxbf = self.fc_rxbf(x)
        
        x_txbf = torch.tanh(x_txbf)
        x_rxbf = torch.tanh(x_rxbf)

        return x_txbf, x_rxbf

    def get_action(self, state):
        x_txbf, x_rxbf = self.forward(state)

        act_txbf = self.softmax_txbf(x_txbf)
        act_rxbf = self.softmax_rxbf(x_rxbf)
        
        return act_txbf, act_rxbf
        






