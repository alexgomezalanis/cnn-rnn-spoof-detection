from __future__ import print_function, division
import os
import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CNN_RNN_Dataset
from utils.collate import collate
import torch.nn as nn
from sklearn.metrics import confusion_matrix


rootPath = os.getcwd()


def eval(args, model, optimizer, device, model_location):
  criterion_dev = nn.CrossEntropyLoss(reduction='sum')

  if args.is_googleColab:
    root_dir = '/content/drive/My Drive/database'
    csv_dir = '/content/cnn-rnn-spoof-detection/protocols'
  else:
    root_dir = './database'
    csv_dir = './protocols'
  
  test_dataset = CNN_RNN_Dataset(
    csv_file=csv_dir + '/' + args.csv_test + '.csv',
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    dataset='test',
    num_classes=args.num_classes)

  test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
    num_workers=args.num_data_workers, collate_fn=collate)

  dev_accuracy, dev_loss = test_epoch(model, device, test_loader, criterion_dev)

  #-----validacion-----------
  outfile = model_location + '/cmTest-' + args.csv_test
  cm, labels_cm = generate_confusion_matrix(model,test_loader,device)
  np.save(outfile,cm)
  outfile = model_location + '/cmTest-' + args.csv_test + '-labels_cm'
  np.save(outfile,labels_cm)

def test_epoch(model, device, data_loader, criterion):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      stfts = batch[0]
      targets = torch.stack(batch[1])
      targets = targets.to(device)
      data = model(stfts)
      data = data.to(device)
      test_loss += criterion(data, targets).item() # sum up batch loss
      pred = data.max(1)[1] # get the index of the max probability
      correct += pred.eq(targets).sum().item()
  #----UNA VEZ QUE TERMINA LA EPOCA DE ENTRENAMIENTO------------------------
  test_loss /= len(data_loader.dataset)
  test_accuracy = 100. * correct / len(data_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(data_loader.dataset), test_accuracy))
  return test_accuracy, test_loss

def get_labels_used_in_cm(all_labels,pred):
  return np.unique(np.concatenate((np.unique(all_labels.numpy()),np.unique(pred.numpy()))))


def generate_confusion_matrix(model,prediction_loader,device):
  with torch.no_grad():
    train_preds, all_labels = get_all_preds(model, prediction_loader,device)
    pred = train_preds.max(1)[1] # get the index of the max probability
    cm = confusion_matrix(all_labels.cpu(),pred.cpu())
    labels_cm =get_labels_used_in_cm(all_labels.cpu(),pred.cpu())
  return cm, labels_cm



def get_all_preds(model, loader,device):
  all_preds = torch.tensor([]).to(device)
  all_labels = torch.tensor([],dtype=torch.long).to(device)
  for batch in loader:
    stfts = batch[0]
    targets = torch.stack(batch[1])
    targets = targets.to(device)
    preds = model(stfts)
    preds = preds.to(device)
    all_preds = torch.cat((all_preds,preds),dim=0)
    all_labels = torch.cat((all_labels,targets),dim=0)
  return all_preds, all_labels
 
  