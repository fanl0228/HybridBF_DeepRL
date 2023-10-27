
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append('../')
from model_backbone.cnn import cnn3d
from model_backbone import ResNet
from dataloader import OfflineDataset
from dataloader.h5py_opts import read_h5py_file

import pdb



class BackboneRL(nn.Module):
    def __init__(self, 
                model_depth=50, 
                nDoppler=128, 
                output_dim=2048):   # [txbf, rxbf]
        super().__init__()
        self.resnet = ResNet.generate_model(model_depth=model_depth,
                                            n_input_channels=nDoppler,
                                            shortcut_type='B',
                                            conv1_t_size=5,
                                            conv1_t_stride=1,
                                            no_max_pool=False,
                                            widen_factor=1.0)
        
    '''
        input data: [batch, channel, Deepth, Height, width]
        channel   --->  Doppler
        Deepth    --->  Frame
        Height    --->  Range
        Width     --->  Antenne
    '''
    def forward(self, x):

        out= self.resnet(x)

        return out


if __name__=="__main__":

    """
    3D resnet
    model_depth = [10, 18, 34, 50, 101, 152, 200]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BackboneRL(model_depth=50, nDoppler=128, output_dim=2048)
    if torch.cuda.is_available():
        model.cuda()

    print(model)
    
    train_dataset = "/home/hx/fanl/HybridBF_DeepRL/datasets/train"
    dataset = OfflineDataset(train_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # dataloader
    for sample_path in dataloader:
        data_dict = read_h5py_file(sample_path[0])
        
        for bIdex in range(data_dict['observationsRA'].shape[0]):
            cube_FRDA = data_dict['observationsRD'][bIdex]   # [frame, range, Doppler, ant] = [10, 64, 128, 16]
            
            # to tensor
            cube_FRDA = torch.from_numpy(10*np.log10(abs(cube_FRDA) ) )
            # [frame, range, Doppler, ant] --> [Doppler, frame, range, ant]
            cube_FRDA = cube_FRDA.permute(2, 0, 1, 3)
            # add batch dim
            cube_FRDA = torch.unsqueeze(cube_FRDA, 0)
            cube_FRDA = cube_FRDA.to(device, non_blocking=True)  

            # cube_frame_RD = 10*np.log10(abs(np.mean(cube_FRDA, -1) ))  # get Frame_RD  [10, 64, 128]
            # cube_ant_RD = np.mean(cube_FRDA, 1)    # get RD ant    [64, 128, 16]
            
            # cube_frame_RD = torch.from_numpy(cube_frame_RD)
            # cube_ant_RD = torch.from_numpy(cube_ant_RD)
            
            # cube_frame_RD = torch.unsqueeze(torch.unsqueeze(cube_frame_RD, 0), 0)        # [1, 1, 10, 64, 128]
            # cube_ant_RD   = torch.unsqueeze(torch.unsqueeze(cube_ant_RD,   0), 0)        # [1, 1, 64, 128, 16] 
            
            out = model(cube_FRDA)   # Resnet [1, 2048]
            #out = model(cube_frame_RD)   # Resnet [1, 2048]
            print("bIdex:{}, out size:{}, out:{} ".format(bIdex, out.size(), out))
            
            
            # pdb.set_trace()









