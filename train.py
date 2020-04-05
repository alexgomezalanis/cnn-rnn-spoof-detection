import os
import sys
import torch #The top-level PyTorch package and tensor library.
import torch.optim as optim #A subpackage that contains standard optimization operations like SGD and Adam.
import torch.nn as nn #A subpackage that contains modules and extensible classes for building neural networks.
import torch.nn.functional as F #A functional interface that contains typical operations used for building neural networks like loss functions, activation functions, and convolution operations.
import numpy as np
from torch.utils.data import DataLoader #A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier
from dataset import CNN_RNN_Dataset
from utils.checkpoint import load_checkpoint
from utils.collate import collate

rootPath = os.getcwd()

def train(args, model, start_epoch, accuracy, criterion, optimizer, device, model_location):
  criterion_dev = nn.CrossEntropyLoss(reduction='sum')

  train_protocol = 'ConjuntoDatosEntrenamiento.csv'
  dev_protocol = 'ConjuntoDatosValidacion.csv'
  root_dir = './database'
  print('num_frame: ',args.num_frames,'num_filts: ',args.num_filts,'window_lenght: ', args.window_length,'frame_shift: ',args.frame_shift)
  train_dataset = CNN_RNN_Dataset(
    csv_file='./protocols/' + train_protocol,
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    dataset='training',
    num_classes=args.num_classes)

  dev_dataset = CNN_RNN_Dataset(
    csv_file='./protocols/' + dev_protocol,
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    dataset='development',
    num_classes=args.num_classes)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_data_workers, collate_fn=collate)

  dev_loader = DataLoader(dev_dataset, batch_size=args.test_batch_size, shuffle=False,
    num_workers=args.num_data_workers, collate_fn=collate)

  numEpochsNotImproving = 0
  best_acc = accuracy
  epoch = start_epoch
  while (numEpochsNotImproving < args.epochs):
    train_epoch(epoch, args, model, device, train_loader, optimizer, criterion)
    epoch += 1
    dev_accuracy, dev_loss = test_epoch(model, device, dev_loader, criterion_dev)
    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'losslogger': dev_loss,
      'accuracy': dev_accuracy
    }
    torch.save(state, model_location + '/epoch-' + str(epoch) + '.pt')
    if (dev_accuracy > best_acc):
      best_acc = dev_accuracy
      numEpochsNotImproving = 0
      torch.save(state, model_location + '/best.pt')
    else:
      numEpochsNotImproving += 1

def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion):
  model.train()
  pid = os.getpid()
  for batch_idx, sample in enumerate(data_loader):
    #(stft, target, _) = sample
    #target = torch.LongTensor(target).to(device)

    optimizer.zero_grad()
    output = model(sample)
    data = output[0]
    target = output[1]
    loss = criterion(data, target)
    optimizer.zero_grad() # zero the parameter gradients
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))
      sys.stdout.flush()

def test_epoch(model, device, data_loader, criterion):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for batch_idx, sample in enumerate(data_loader):
      #(stft, target, _) = sample
      #target = torch.LongTensor(target).to(device)
      output = model(sample)
      data = output[0]
      target = output[1]
      test_loss += criterion(data, target).item() # sum up batch loss
      pred = data.max(1)[1] # get the index of the max probability
      correct += pred.eq(target).sum().item()

  test_loss /= len(data_loader.dataset)
  test_accuracy = 100. * correct / len(data_loader.dataset)
  print('\nDevelopment set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(data_loader.dataset), test_accuracy))

  return test_accuracy, test_loss