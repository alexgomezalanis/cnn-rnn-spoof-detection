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
import torch.nn.functional as F


rootPath = os.getcwd()


def eval(args, model, optimizer, device, model_location):
  criterion_dev = nn.CrossEntropyLoss(reduction='sum')

  if args.is_googleColab:
    root_dir = '/content/drive/My Drive/database'
    csv_dir = '/content/cnn-rnn-spoof-detection/protocols'
  else:
    root_dir = './database'
    csv_dir = './protocols'

  if args.eval_mezcla:
    dataset = 'mezcla'
  else:
    dataset = 'test'
  
  test_dataset = CNN_RNN_Dataset(
    csv_file=csv_dir + '/' + args.csv_test + '.csv',
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    dataset=dataset,
    num_classes=args.num_classes)

  test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
    num_workers=args.num_data_workers, collate_fn=collate)

  dev_accuracy, dev_loss,all_preds, all_labels = test_epoch(model, device, test_loader, criterion_dev)

  #------calculamos el accuracy ponderado del conjunto de test seleccionado-----
  accuracy_ponderado = calculo_accuracy_ponderado(all_labels,all_preds)
  print('Accuracy Ponderado: ',accuracy_ponderado)

  #-----cm test and asociate labels-----------
  print('calculando la matriz de confusion del conjunto de test... \n')
  outfile = model_location + '/cmTest-' + args.csv_test
  cm = confusion_matrix(all_labels.cpu(),all_preds.cpu())
  labels_cm =get_labels_used_in_cm(all_labels.cpu(),all_preds.cpu())
  np.save(outfile,cm)
  outfile = model_location + '/cmTest-' + args.csv_test + '-labels_cm'
  np.save(outfile,labels_cm)

def test_epoch(model, device, data_loader, criterion):
  model.eval()
  test_loss = 0
  correct = 0
  all_preds = torch.tensor([],dtype=torch.long).to(device)
  all_labels = torch.tensor([],dtype=torch.long).to(device)
  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      stfts = batch[0]
      targets = torch.stack(batch[1])
      targets = targets.to(device)
      names = batch[2]
      data = model(stfts)
      data = data.to(device)
      pSumadas,estimacionMMSE= MMSE(data,device)
      test_loss += criterion(data, targets).item() # sum up batch loss
      pred = data.max(1)[1] # get the index of the max probability
      correct += pred.eq(targets).sum().item()
      all_preds = torch.cat((all_preds,pred),dim=0)
      all_labels = torch.cat((all_labels,targets),dim=0)
  #----UNA VEZ QUE TERMINA LA EPOCA DE ENTRENAMIENTO------------------------
  test_loss /= len(data_loader.dataset)
  test_accuracy = 100. * correct / len(data_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(data_loader.dataset), test_accuracy))

  return test_accuracy, test_loss, all_preds, all_labels

def get_labels_used_in_cm(all_labels,pred):
  return np.unique(np.concatenate((np.unique(all_labels.numpy()),np.unique(pred.numpy()))))


def get_new_classes(all_labels,all_preds,device):

  mapping = {
  0: [0],
  1: [1,2],
  2: [2,1],
  3: [3],
  4: [4],
  5: [5],
  6: [6,7],
  7: [7,6],
  8: [8],
  9: [9],
  10: [10],
  11: [11] }

  newClasses = { 0:0, 1:1, 2:1, 3:2, 4:3,
                5:4,6:5,7:5,8:6,
                9:7,10:8,11:9 }

  # xx = { 'limpio':0,
  #       'clippingBajo':1, 'clippingMedio':2, 'clippingAlto':3,
  #       'reverberacionBaja':4, 'reverberacionMedia':5, 'reverberacionAlta':6,
  #       'SNRalto':7,'SNRmedio':8,'SNRbajo':9 }

  new_labels = []
  new_preds = []
  correct = 0
  size = all_labels.size()[0]
  size2 = all_preds.size()[0]
  for i, label in enumerate(all_labels):
    label = label.item()
    pred = all_preds[i].item()
    if pred in mapping[label]: 
      correct += 1
    new_labels.append(newClasses[label])
    new_preds.append(newClasses[pred])
  #calculamos el nuevo accuracy
  accuracy = correct/size
  return accuracy, torch.tensor(new_labels).to(device),torch.tensor(new_preds).to(device)



def calculo_accuracy_ponderado(all_labels,all_preds):
    mapping = {
      0: [0,1],
      1: [1,0,2],
      2: [2,1,3],
      3: [3,2,4],
      4: [4,3],
      5: [5,6],
      6: [6,5,7],
      7: [7,6,8],
      8: [8,7],
      9: [9,10],
      10: [10,9,11],
      11: [11,10] }

    
    # LIST_CLASSES = ('0limpio','1clipping_09_5_percent','2clipping_5_10_percent','3clipping_20_40_percent','4clipping_40_70_percent',
    #                 '5reverberacion_Lowhight','6reverberacion_MediumLow','7reverberacion_MediumHight','8reverberacion_Hight',
    #                 '9noise_0_5db_SNR','10noise_5_10db_SNR','11noise_10_20db_SNR')
  
    correct = 0
    size_allLabels = all_labels.size()[0]
    for i, label in enumerate(all_labels):
      label = label.item()
      pred = all_preds[i].item()
      if pred in mapping[label]: 
        if mapping[label].index(pred) == 0: #es la clase 100%
          correct += 1
        else: #es una clase adjacente
          correct +=0.5
    #calculamos el nuevo accuracy
    accuracy_ponderado = correct/size_allLabels
    return accuracy_ponderado



def MMSE(data,device):

  prob_locuciones = F.softmax(data,dim=1)
  numero_locuciones = prob_locuciones.shape[0]

  estimaciones = []
  probabilidadesSumadasPorDistorsion = []
  for i in range(numero_locuciones):
    clipping = prob_locuciones[i][1] + prob_locuciones[i][2] + prob_locuciones[i][3] + prob_locuciones[i][4]
    rever = prob_locuciones[i][5] + prob_locuciones[i][6] + prob_locuciones[i][7] + prob_locuciones[i][8]
    noise = prob_locuciones[i][9] + prob_locuciones[i][10] + prob_locuciones[i][11]
    limpio = prob_locuciones[i][0]

    estimacionNoise = prob_locuciones[i][9] * 2.5 + prob_locuciones[i][10] * 7.5 + prob_locuciones[i][11] * 15 + (clipping + limpio + rever) * 25
    estimacionClipping = prob_locuciones[i][1] * 1 + prob_locuciones[i][2] * 2 + prob_locuciones[i][3] * 3 + prob_locuciones[i][4] * 4 + (noise + limpio + rever) * 0
    estimacionRever = prob_locuciones[i][5] * 1 + prob_locuciones[i][6] * 2 + prob_locuciones[i][7] * 3 + prob_locuciones[i][8] * 4 + (noise + limpio + clipping) * 0
 
    estimacion = [estimacionClipping,estimacionRever,estimacionNoise]
    estimaciones.append(estimacion)
    probabilidadesSumadas = [limpio,clipping,rever,noise]
    probabilidadesSumadasPorDistorsion.append(probabilidadesSumadas)

  return torch.tensor(estimaciones), torch.tensor(probabilidadesSumadasPorDistorsion)