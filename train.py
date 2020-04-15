import os
import sys
import torch #The top-level PyTorch package and tensor library.
import torch.optim as optim #A subpackage that contains standard optimization operations like SGD and Adam.
import torch.nn as nn #A subpackage that contains modules and extensible classes for building neural networks.
import torch.nn.functional as F #A functional interface that contains typical operations used for building neural networks like loss functions, activation functions, and convolution operations.
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader #A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier
from dataset import CNN_RNN_Dataset
from utils.checkpoint import load_checkpoint
from utils.collate import collate
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

rootPath = os.getcwd()

def train(args, model, start_epoch, accuracy, criterion, optimizer, device, model_location):
  criterion_dev = nn.CrossEntropyLoss(reduction='sum')

  train_protocol = 'ConjuntoDatosEntrenamiento.csv'
  dev_protocol = 'ConjuntoDatosValidacion.csv'

  if args.is_googleColab:
    root_dir = '/content/drive/My Drive/database'
    csv_dir = '/content/cnn-rnn-spoof-detection/protocols'
  else:
    root_dir = './database'
    csv_dir = './protocols'
  

  train_dataset = CNN_RNN_Dataset(
    csv_file= csv_dir + '/' + train_protocol,
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    dataset='training',
    num_classes=args.num_classes)

  dev_dataset = CNN_RNN_Dataset(
    csv_file=csv_dir + '/' + dev_protocol,
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
  #pintamos la matrix de confusión en la última epoca 
  #------entrenamiento------
  cm = generate_confusion_matrix(model,train_loader,device)
  plt.figure(figsize=(args.num_classes,args.num_classes))
  plot_confusion_matrix(cm,train_dataset.classes,title='Train Confusion matrix')

  #-----validacion-----------
  cm = generate_confusion_matrix(model,dev_loader,device)
  plt.figure(figsize=(args.num_classes,args.num_classes))
  plot_confusion_matrix(cm,train_dataset.classes,title='Validation Confusion matrix')

def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion):
  model.train()
  correct = 0
  train_loss=0
  pid = os.getpid()
  for batch_idx, sample in enumerate(data_loader):
    output = model(sample)
    data = output[0].to(device)
    target = output[1].to(device)
    loss = criterion(data, target)
    optimizer.zero_grad() # zero the parameter gradients
    loss.backward()
    optimizer.step()
    #calculamos el numero de errores en cada batch
    pred = data.max(1)[1] # get the index of the max probability
    correct += pred.eq(target).sum().item()
    train_loss += loss
    if batch_idx % args.log_interval == 0:
      print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))
      sys.stdout.flush()
  #datos a mostrar al terminar una epoca de entrenamiento
  test_accuracy = 100. * correct / len(data_loader.dataset)
  print('\tTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(data_loader.dataset), test_accuracy))


def test_epoch(model, device, data_loader, criterion):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad(): #indicamos que no hay que tener en cuenta el calculo de gradiente (Desactivamos)
    for batch_idx, sample in enumerate(data_loader):
      output = model(sample)
      data = output[0].to(device)
      target = output[1].to(device)
      test_loss += criterion(data, target).item() # sum up batch loss
      pred = data.max(1)[1] # get the index of the max probability
      correct += pred.eq(target).sum().item()

  test_loss /= len(data_loader.dataset)
  test_accuracy = 100. * correct / len(data_loader.dataset)
  print('\nDevelopment set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(data_loader.dataset), test_accuracy))

  return test_accuracy, test_loss

 
def generate_confusion_matrix(model,prediction_loader,device):
  with torch.no_grad():
    train_preds, all_labels = get_all_preds(model, prediction_loader,device)
    pred = train_preds.max(1)[1] # get the index of the max probability
  
  return confusion_matrix(all_labels,pred)



def get_all_preds(model, loader,device):
  all_preds = torch.tensor([]).to(device)
  all_labels = torch.tensor([],dtype=torch.long).to(device)
  for batch in loader:
    output = model(batch)
    preds = output[0].to(device)
    target = output[1].to(device)
    all_preds = torch.cat((all_preds,preds),dim=0)
    all_labels = torch.cat((all_labels,target),dim=0)
  return all_preds, all_labels