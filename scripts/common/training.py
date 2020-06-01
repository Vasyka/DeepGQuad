import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb

from . import metrics
from . import utils

# Training
def train(net, epochs, optimizer, train, val = None, batch_size = 32, params = {}, eval_epochs = 2, use_wandb = True, 
          new_run = True, save_model = False, all_losses = None, all_accs = None):
    criterion = nn.BCELoss()
    #criterion = nn.SmoothL1Loss()
    best_val_loss = 10000
    sequence_case = False
    device = utils.get_device()
    model_losses = []
    model_accs = []

    # Get dataloaders
    train_loader = DataLoader(train, batch_size=batch_size,shuffle=True)
    if val:
        val_loader = DataLoader(val, batch_size=batch_size,shuffle=True)
    
    if use_wandb and new_run:
        wandb.init(project="gquad_hybrid2")
        wandb.watch(net)
        params['epochs'] = epochs
        wandb.config.update(params)

    for epoch in range(epochs):  
        running_loss = 0.0
        running_acc = 0.0
        running_iou = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].type(torch.FloatTensor).unsqueeze(1).to(device) 

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            #print(outputs.size(),labels.size())
            loss = criterion(outputs, labels)
            acc = metrics.binary_acc(outputs, labels)
            
            # sequence case
            if labels.shape[1] > 1:
                sequence_case = True
                iou = metrics.inter_over_union(outputs, labels)
                #loss = dice_coef(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            running_acc += acc.item()
            if sequence_case:
                running_iou += iou.item()
            if i % 100 == 99: 
                output_metrics = 'Epoch: %d, step: %4d | loss: %.3f | acc: %.3f ' % \
                      (epoch + 1, i + 1, running_loss / 100, running_acc / 100) 
                if sequence_case:
                    output_metrics += ('| iou  %.3f' % (running_iou / 100))
                print(output_metrics)
                if use_wandb:
                    wandb.log({"Train Loss": running_loss / 100, 
                               "Train Accuracy": running_acc / 100,
                               "Train IoU": running_iou / 100})
                else:
                    model_losses.append(running_loss / 100)
                    model_accs.append(running_acc / 100)
                running_loss = 0.0
                running_acc = 0.0
                running_iou = 0.0

        # Evaluate on validation part   
        if eval_epochs > 0 and epoch % eval_epochs == eval_epochs - 1:
          val_loss = 0.0
          val_iou = 0.0
          val_acc = 0.0
          print('Validation:')
          flag = False
          with torch.no_grad():
            for val_i, data in enumerate(val_loader, 0):
              inputs, labels = data[0].to(device), data[1].type(torch.FloatTensor).unsqueeze(1).to(device) #non_blocking=True)
              outputs = net(inputs)

              #val_loss += dice_coef(outputs, labels, criterion)
              val_loss += criterion(outputs, labels,)
              val_acc += metrics.binary_acc(outputs, labels)
              val_iou += metrics.inter_over_union(outputs, labels)
                
            val_iou /= (val_i + 1)
            val_acc /= (val_i + 1)
            val_loss /= (val_i+1)
            print('Validation loss: %.3f | Validation iou: %.3f | Validation Accuracy %.3f' %
                        ( val_loss, val_iou, val_acc))
            if use_wandb:
                wandb.log({"Validation Loss": val_loss, 
                            "Validation Accuracy": val_acc,
                            "Validation IoU": val_iou})

            # Check loss and save model if it is the best
            if best_val_loss >= val_loss:
              best_val_loss = val_loss
              print('Best validation loss: %.3f' % (best_val_loss))
              if save_model:
                if use_wandb:
                    print(f'Saving model to ./results/Hybrid models/{wandb.run.name}.pth.tar.')
                    exp_name = wandb.run.save()
                    torch.save({'state_dict': net.state_dict()}, f'./results/Hybrid models/{wandb.run.name}.pth.tar')
    print('Finished Training')
    if eval_epochs > 0:
        print('Best validation loss: %.3f' % (best_val_loss))
    if not use_wandb:
        all_losses.append(model_losses)
        all_accs.append(model_accs)   
        return all_losses, all_accs

# Evaluation on test part
def evaluation(net, test_dataset, batch_size = 32):
    y_pred_list = []
    y_labels_list=[]
    sequence_case=False
    device = utils.get_device()
    iou=0.0
    acc=0.0
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            
            if len(labels.size()) == 1:
                # binary case
                sequence_case = False
                outputs = torch.round(outputs)
            else:
                # sequence case
                sequence_case = True
                iou += metrics.inter_over_union(outputs, labels)
                acc += metrics.binary_acc(outputs, labels.unsqueeze(1))
            y_pred_list.append(outputs.cpu().numpy())
            y_labels_list.append(labels.cpu().numpy())
    if sequence_case:     
        iou = iou / (i + 1)
        acc = acc / (i + 1)
        print('IoU: %.3f, acc %.3f' %(iou, acc))
    else:
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        y_labels_list = [a.squeeze().tolist() for a in y_labels_list]
    return y_labels_list, y_pred_list

# Load saved model
def load_model(net, path):
    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location='cpu')
        device = 'cpu'
    else:
        checkpoint = torch.load(path)
        device = 'cuda:0'
    print("Model's state_dict:")
    for param_tensor in checkpoint['state_dict']:
        print(param_tensor, "\t", checkpoint['state_dict'][param_tensor].size())
    net.load_state_dict(checkpoint['state_dict'])
    print(net)
    return net.to(device)
