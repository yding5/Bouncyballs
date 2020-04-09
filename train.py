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
              epochs=20,
              batch_size=16,
              lr=0.001,
              save_cp=True,
              data_folder='data',
              save_pred=None):

    train_dataset = Dataset_Traj(data_folder, 0, 10000, aug=False)
    test_dataset = Dataset_Traj(data_folder, 10000, 11000, aug=False)

    n_train = len(train_dataset)
    #print(n_train)
    n_test = len(test_dataset)
    #print(n_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    
    writer = SummaryWriter(comment=f'Traj_LR_{lr}_BS_{batch_size}_Epoch_{epochs}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Save test images:  {save_pred}
    ''')
        #Training size:   {n_train}
        #Validation size: {n_val}

    optimizer = optim.Adam(net.parameters(), lr=lr)

    criterion = nn.MSELoss(reduction='none')

    
    def eval_and_print():
        # train set
        train_res = eval_net(net, train_loader, device)
        train_loss = train_res['avg_loss']
        train_loss_over_time = train_res['loss_over_time']
        logging.info('Train Loss: {}'.format(train_loss))
        #logging.info('Train Loss over time: {}'.format(train_loss_over_time))
        # test set
        test_res = eval_net(net, test_loader, device)
        test_loss = test_res['avg_loss']
        test_loss_over_time = test_res['loss_over_time']
        logging.info('Test Loss: {}'.format(test_loss))
        #logging.info('Test Loss over time: {}'.format(test_loss_over_time))

    for epoch in range(epochs):
        print(f'starting epoch {epoch}')
        eval_and_print()
        
        net.train()
        epoch_loss = 0
        #with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            net_input = batch['net_input']
            target = batch['net_output']
            mask = batch['valid_mask']
   
            net_input = net_input.to(device=device, dtype=torch.float32)
            target = target.to(device=device)
            mask = mask.to(device=device, dtype=torch.float32)

            pred = net(net_input)

            loss = criterion(pred, target)
            # block loss for invalid prediction
            loss = torch.mean(loss * mask)

            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
                
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    
    eval_and_print()

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--data', dest='data', type=str, default='/hdd_d/yukun/SWIM/IN/dataset/level_1/',
                        help='Load pred data from the folder in data/pred/')
    parser.add_argument('-s', '--save-pred', dest='savepred', type=str, default=None,
                        help='save model prediction')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # define model
    net = LSTM_Init_To_Many()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    
    dir_checkpoint = 'checkpoints/'

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  save_pred=args.savepred,
                  data_folder=args.data,
                 )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
