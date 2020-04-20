import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CNN_RNN_Dataset
from utils.checkpoint import load_checkpoint
from utils.collate import collate
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter

rootPath = os.getcwd()

def train(args, model, start_epoch, accuracy, numEpochsNotImproving, criterion, optimizer, device, model_location):
  criterion_dev = nn.CrossEntropyLoss(reduction='sum')
  tb = SummaryWriter()
  globaliter = 0
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

  accuracy_vector_train = []
  accuracy_vector_val = []
  loss_vector_val = []
  loss_vector_train = []

  numEpochsNotImproving = numEpochsNotImproving
  best_acc = accuracy
  epoch = start_epoch
  accuracy_train_per_epoch = []
  while (numEpochsNotImproving < args.epochs):
    epoch += 1
    train_accuracy, train_loss = train_epoch(epoch, args, model, device, train_loader, optimizer, criterion,tb)
    dev_accuracy, dev_loss = test_epoch(model, device, dev_loader, criterion_dev,tb,epoch)
    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'losslogger': dev_loss,
      'accuracy': dev_accuracy,
      'numEpochsNotImproving': numEpochsNotImproving
    }
    torch.save(state, model_location + '/epoch-' + str(epoch) + '.pt')
    accuracy_vector_train.append(train_accuracy)
    accuracy_vector_val.append(dev_accuracy)
    loss_vector_val.append(dev_loss)
    if (dev_accuracy > best_acc):
      best_acc = dev_accuracy
      numEpochsNotImproving = 0
      torch.save(state, model_location + '/best.pt')
      #guardamos la matriz de confusion de entrenamiento y validacion
        #------entrenamiento------
      outfile = model_location + '/cmTrain-best-' + str(epoch)
      cm = generate_confusion_matrix(model,train_loader,device)
      np.save(outfile,cm)
        #-----validacion-----------
      outfile = model_location + '/cmValidation-best-' + str(epoch)
      cm = generate_confusion_matrix(model,dev_loader,device)
      np.save(outfile,cm)
    else:
      numEpochsNotImproving += 1
  #cuando el entrenamiento termine se guardan los vectores con los resultados:
    #------train--------
  np.save(model_location + '/vAccuracyTrain',np.array(accuracy_vector_train))
  np.save(mode_location + '/vLossTrain',np.array(loss_vector_train))
    #------val--------
  np.save(model_location + '/vAccuracyVal',np.array(accuracy_vector_val))
  np.save(mode_location + '/vLossVal',np.array(loss_vector_val))

def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion,tb):
  model.train()
  correct = 0
  train_loss=0
  pid = os.getpid()
  for batch_idx, batch in enumerate(data_loader):
    stfts = batch[0]
    targets = torch.stack(batch[1])
    targets = targets.to(device)
    data = model(stfts)
    data = data.to(device)
    loss = criterion(data, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #-------SE CALCULA EL NUMERO DE ERRORES DE CADA BATCH Y SE VA ACOMULANDO---------------
    pred = data.max(1)[1] # get the index of the max probability
    correct += pred.eq(targets).sum().item()
    train_loss += loss.item()
    if batch_idx % args.log_interval == 0:
      print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))
      sys.stdout.flush()
      tb.add_scalar('Loss/train', loss.item(),(batch_idx /len(data_loader))+epoch-1)
  #----UNA VEZ QUE TERMINA LA EPOCA DE ENTRENAMIENTO------------------------
  train_loss /= len(data_loader.dataset)
  train_accuracy = 100. * correct / len(data_loader.dataset)
  print('\tTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(data_loader.dataset), train_accuracy))
      #-------enviamos los resultados a tensorboard-------------------
  tb.add_scalar('Accuracy/train', train_accuracy, epoch)

  return train_accuracy, train_loss
  

def test_epoch(model, device, data_loader, criterion,tb,epoch):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad(): #indicamos que no hay que tener en cuenta el calculo de gradiente (Desactivamos)
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
  print('\nDevelopment set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(data_loader.dataset), test_accuracy))
        #-------enviamos los resultados a tensorboard-------------------
  tb.add_scalar('Loss/val', test_loss,epoch)
  tb.add_scalar('Accuracy/val', test_accuracy, epoch)
  return test_accuracy, test_loss

 
def generate_confusion_matrix(model,prediction_loader,device):
  with torch.no_grad():
    train_preds, all_labels = get_all_preds(model, prediction_loader,device)
    pred = train_preds.max(1)[1] # get the index of the max probability
    mc = confusion_matrix(all_labels.cpu(),pred.cpu())
  return mc



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
