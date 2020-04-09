import torch
import random
import matplotlib.pyplot as plt
import math
from matplotlib import animation, rc
from IPython.display import HTML, Image

def eval_net(net, data_loader, device):
    criterion = torch.nn.MSELoss(reduction='none')
    net.eval()
    loss_list = []
    loss_over_time_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            net_input = batch['net_input']
            target = batch['net_output']
            mask = batch['valid_mask']


            net_input = net_input.to(device=device, dtype=torch.float32)
            target = target.to(device=device)
            mask = mask.to(device=device, dtype=torch.float32)

            pred = net(net_input)

            raw_loss = criterion(pred, target)
            # block loss
            loss = torch.mean(raw_loss * mask)
            #print(raw_loss.shape)
            loss_over_time = torch.mean(raw_loss * mask, dim=2)
            #print(loss_over_time.shape)
            loss_over_time_list.append(loss_over_time)
            loss_list.append(loss.item())

    avg_loss = sum(loss_list)/len(loss_list)
    loss_over_time = torch.cat(loss_over_time_list, dim=0)
    loss_over_time = torch.mean(loss_over_time, dim=0)
    #print(loss_over_time.shape)
    eval_res = {'avg_loss': avg_loss, 'loss_over_time': loss_over_time}
    return eval_res




def vis_results(net, dataset, data_loader, device, save_path='figures/level1/', n_samples=20):
    criterion = torch.nn.MSELoss(reduction='none')
    net.eval()
    pred_list = []
    label_list = []
    net_input_list = []
    

    with torch.no_grad():
        for batch in data_loader:
            net_input = batch['net_input']
            target = batch['net_output']
            mask = batch['valid_mask']

            net_input = net_input.to(device=device, dtype=torch.float32)
            target = target.to(device=device)
            mask = mask.to(device=device, dtype=torch.float32)

            pred = net(net_input)
            pred_list.append(pred.cpu())
            net_input_list.append(net_input.cpu())
            label_list.append(target.cpu())
            
          
    all_pred = torch.cat(pred_list, dim=0)
    all_net_input = torch.cat(net_input_list, dim=0)
    all_label = torch.cat(label_list, dim=0)
    #print(all_net_input.shape)  
    
    idxs = random.choices(list(range(0,all_pred.shape[0])), k=n_samples)
    print(f'random selected samples: {idxs}')
    
    vis_pred = all_pred[idxs].numpy()
    vis_input = all_net_input[idxs].numpy()
    vis_label = all_label[idxs].numpy()

    assert vis_pred.shape == (n_samples, all_pred.shape[1], all_pred.shape[2])
    
    # de-normalize
    vis_pred = (vis_pred * dataset.label_std) + dataset.label_mean 
    vis_label = (vis_label * dataset.label_std) + dataset.label_mean 
    vis_input = (vis_input * dataset.in_std) + dataset.in_mean 
    print(vis_input.shape)
    
    # line property
    len_list = dataset.len_list
    theta_list = dataset.theta_list
    
    def vis_traj(label_seq, pred_seq, init_input, len_list, theta_list, name):
        fig = plt.figure(figsize=(8.0, 8.0))
        invalid_step_pred = label_seq.shape[0]
        for step in range(label_seq.shape[0]):
            if (pred_seq[step,0] < 50) or (pred_seq[step,0] > 550) or (pred_seq[step,1] < 50) or (pred_seq[step,1] > 550):
                invalid_step_pred = step
                break
                
        plt.scatter(pred_seq[:invalid_step_pred,0],pred_seq[:invalid_step_pred,1], label='pred', s=40, alpha=0.5)
        plt.scatter(pred_seq[invalid_step_pred:,0],pred_seq[invalid_step_pred:,1], label='pred_invalid', s=40, alpha=0.5)
        plt.scatter(label_seq[:,0],label_seq[:,1], label='real', s=40, alpha=0.5)
        for i, length in enumerate(len_list):
            plt.plot((init_input[2*i], init_input[2*i]+math.cos(theta_list[i])*length),(init_input[2*i+1], init_input[2*i+1]+math.sin(theta_list[i])*length), 'k', label='platform', linewidth=5)
        
        plt.legend()
        plt.xlim([0,600])
        plt.ylim([0,600])
        fig.savefig(save_path+'{}.png'.format(name))
        return
    
    def save_as_gif(label_seq, pred_seq, init_input, len_list, theta_list, name):
        fig = plt.figure(figsize=(8.0, 8.0))
        ax = plt.axes()
        
        lines = []
        lines_x = []
        lines_y = []
        line_real, = ax.plot([],[], 'o', markersize = 12, alpha=0.5, label='real')
        line_pred, = ax.plot([],[], 'o', markersize = 12, alpha=0.5, label='pred')
        
        for i, length in enumerate(len_list):
            plt.plot((init_input[2*i], init_input[2*i]+math.cos(theta_list[i])*length),(init_input[2*i+1], init_input[2*i+1]+math.sin(theta_list[i])*length), 'k', label='platform {}'.format(i), linewidth=5)
        
        plt.legend()

        lines.append(line_real)
        lines.append(line_pred)
        lines_x.append([])
        lines_y.append([])
        lines_x.append([])
        lines_y.append([])
        
        invalid_step_pred = False
        
        def init():
            for line in lines:
                line.set_data([],[])
            return lines
        
        def animate(i):
            nonlocal invalid_step_pred
            lines_x[0].append(label_seq[i, 0])
            lines_y[0].append(label_seq[i, 1])
            if not invalid_step_pred:
                if (pred_seq[i,0] < 40) or (pred_seq[i,0] > 560) or (pred_seq[i,1] < 40) or (pred_seq[i,1] > 560):
                    invalid_step_pred = True
            if not invalid_step_pred:
                lines_x[1].append(pred_seq[i, 0])
                lines_y[1].append(pred_seq[i, 1])
                
            lines[0].set_data(lines_x[0], lines_y[0])
            lines[1].set_data(lines_x[1], lines_y[1])
            
            ax.set_xlim([0,600])
            ax.set_ylim([0,600])
                
            return (lines)
        
            
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=pred_seq.shape[0], interval=40, blit=True)
        
        anim.save(save_path+'{}.mp4'.format(name))
        plt.close()
    
    for i in range(vis_pred.shape[0]):
        print(f'process sample {i}')
        vis_traj(vis_label[i], vis_pred[i], vis_input[i], len_list, theta_list, i)
        save_as_gif(vis_label[i], vis_pred[i], vis_input[i], len_list, theta_list, i)
    
    
    
    
    return