import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter
from utils.dataset import *
from utils.model import *
from utils.eval import *
from torch.utils.data import DataLoader, random_split




def train_net(net,
              device,
              dataset,
              data_loader,
              save_path='debug'):


    vis_results(net, dataset, data_loader, device, save_path=save_path, n_samples=20)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--data', dest='data', type=str, default='/hdd_d/yukun/SWIM/IN/dataset/level_1/',
                        help='Load test data from the folder')
    parser.add_argument('-s', '--save-path', dest='savepath', type=str, default='figures/level1/',
                        help='save model prediction')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    net = LSTM_Init_To_Many_3()

    net.load_state_dict(
        torch.load(args.load, map_location=device)
    )
    logging.info(f'Model loaded from {args.load}')

    
    
    net.to(device=device)
    
    
    test_dataset = Dataset_Traj(args.data, 10000, 11000, aug=False)
    n_test = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    try:
        train_net(net=net,
                  dataset=test_dataset,
                  data_loader=test_loader,
                  device=device,
                  save_path=args.savepath
                 )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
