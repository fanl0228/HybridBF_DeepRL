import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.MLP_model import MLP
from agents.backboneRL import BackboneRL

class ValueCritic(nn.Module):
    def __init__(self, 
                state_shape,    # [batch, Doppler, frame, range, ant]
                ) -> None:
        super(ValueCritic, self).__init__()

        self.nDoppler = state_shape[1]

        self.backbone = BackboneRL(model_depth=50, nDoppler=self.nDoppler, output_dim=2048)
        
        # value architecture
        self.l1 = nn.Linear(2048, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        
        feature = self.backbone(state) 

        value = F.relu(self.l1(feature))
        value = F.relu(self.l2(value))
        value = self.l3(value)

        return value


class ActionCritic(nn.Module):

    def __init__(self, 
                state_shape,    # [frame, range, Doppler, ant] = [10, 64, :, 16]
                action_shape,     # [tx/rxbf, beams] = [2, 121]
                ):
        super(ActionCritic, self).__init__()

        self.nDoppler = state_shape[1]
        self.txbf_output_dim = action_shape[0]
        self.rxbf_output_dim = action_shape[1]

        self.backbone_state = BackboneRL(model_depth=50, nDoppler=self.nDoppler, output_dim=2048)  # output [1, 2048]

        self.backbone_action = MLP(2*self.rxbf_output_dim, out_dim=512, hidden_dim=1024, n_layers=3)
        
        
        # Q1 architecture
        self.l1 = nn.Linear(2560, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(2560, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):

        state = self.backbone_state(state)  #[1, 2048]

        action = action.view(1, -1)
        
        action = self.backbone_action(action)  # [1, 512]
        
        feature = torch.cat([state, action], axis=1)

        q1 = F.relu(self.l1(feature))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(feature))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    # def Q1(self, state, action):
    #     sa = torch.cat([state, action], 1)

    #     q1 = F.relu(self.l1(sa))
    #     q1 = F.relu(self.l2(q1))
    #     q1 = self.l3(q1)
    #     return q1


