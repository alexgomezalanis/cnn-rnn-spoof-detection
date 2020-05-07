from __future__ import print_function, division
import argparse
import torch  #The top-level PyTorch package and tensor library.
import os
import torch.nn as nn #	A subpackage that contains modules and extensible classes for building neural networks.
import torch.optim as optim #A subpackage that contains standard optimization operations like SGD and Adam.
import torch.multiprocessing as mp
from utils.checkpoint import load_checkpoint
from utils.create_directory import createDirectory
from model import CNN_RNN
from eval import eval
from train import train

# Training settings
parser = argparse.ArgumentParser(description='CNN RNN audio distortions')
parser.add_argument('--is-googleColab', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='True: train with google Colab// False: train in local')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs for early stopping (default: 2)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=5, metavar='N',
                    help='how many eval processes to use (default: 5)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--num-data-workers', type=int, default=8,
                    help='How many processes to load data') #default=8
parser.add_argument('--num-filts', type=int, default=256,
                    help='How many filters to compute STFT')
parser.add_argument('--num-frames', type=int, default=128,
                    help='How many frames to compute STFT')
parser.add_argument('--n-shift', type=int, default=32,
                    help='')
parser.add_argument('--window-length', type=float, default=0.025,
                    help='Window Length to compute STFT (s)')
parser.add_argument('--frame-shift', type=float, default=0.010,
                    help='Frame Shift to compute STFT (s)')
parser.add_argument('--train', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
parser.add_argument('--eval', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to eval the model')
parser.add_argument('--eval-separately', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='evalua todas las clases del conjunto de test por separado')
parser.add_argument('--eval-mezcla', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='evalua los conjuntos de datos con distorsiones mezcladas')
parser.add_argument('--version', type=str, default='v3',
                    help='Version to save the model')
parser.add_argument('--num-classes', type=int, default=12, metavar='N',
                    help='Number of training classes')
parser.add_argument('--load-epoch', type=int, default=-1,
                    help='Saved epoch to load and start training')
parser.add_argument('--load-trainModel', type=str, default='3',
                    help='path to load train model')
parser.add_argument('--csv-test', type=str, default='0_5db_SNR',
                    help='path to load train model (default: ConjuntoDatosTest)')


def main():
  args = parser.parse_args()
  print(args)

  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

  torch.manual_seed(args.seed)
  # Los valores aleatorios seguirán siendo "aleatorios" pero en un orden definido.
  # Es decir, si reinicia su script, se crearán los mismos números aleatorios.

  mp.set_start_method('spawn')
  model = CNN_RNN(args.num_classes,args.num_frames,args.n_shift,device).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  #path modelSaves to load test model
  if not args.is_googleColab:
    rootPath = os.getcwd()
    model_location = os.path.join(rootPath, 'modelsSave')
  else:
    model_location = os.path.join('/content/drive/My Drive', 'modelsSave')

  if args.train:
      # create dir to save epochs
    if not args.is_googleColab:
      rootPath = os.getcwd()
      model_location = os.path.join(rootPath, 'models')
    else:
      model_location = os.path.join('/content/drive/My Drive', 'models')
    createDirectory(model_location)
    
    if (args.load_epoch != -1):
      path_model_location = os.path.join(model_location, 'epoch-' + str(args.load_epoch) + '.pt')
      model, optimizer, start_epoch, losslogger, accuracy, numEpochsNotImproving = load_checkpoint(model, optimizer, path_model_location)
    else:
      numEpochsNotImproving=0
      start_epoch = 0
      accuracy = 0
    train(
      args=args,
      model=model,
      start_epoch=start_epoch,
      accuracy=accuracy,
      numEpochsNotImproving=numEpochsNotImproving,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
      model_location=model_location)

  if args.eval: #ConjuntoDatosTest
    #del conjunto de datos se puede seleccionar todo al completo o cada clase por separado
    path_model_location = os.path.join(model_location, args.load_trainModel)
    path_model_location = os.path.join(path_model_location,'best.pt')
    model, optimizer, start_epoch, losslogger, accuracy, numEpochsNotImproving = load_checkpoint(model, optimizer, path_model_location)
    print('Empieza el Test:\n')
    eval(
      args=args,
      model=model,
      optimizer=optimizer,
      device=device,
      model_location=model_location)
  
  if args.eval_separately:
    #testea cada clase del conjunto de test por separado
    path_model_location = os.path.join(model_location, args.load_trainModel)
    path_model_location = os.path.join(path_model_location,'best.pt')
    model, optimizer, start_epoch, losslogger, accuracy, numEpochsNotImproving = load_checkpoint(model, optimizer, path_model_location)

    clasesAisladas = ('limpio', '05_09_percent',
      '09_5_percent','5_10_percent','10_20_percent','20_40_percent', '40_70_percent','70_90_percent',
      'LowLow','Lowhight', 'MediumLow', 'MediumHight', 'Hight','HightHight',
      '0_5db_SNR','5_10db_SNR','10_20db_SNR')

    for clase in clasesAisladas:
          print(clase)
          args.csv_test = clase
          eval(
            args=args,
            model=model,
            optimizer=optimizer,
            device=device,
            model_location=model_location)

  if args.eval_mezcla: #accede a las bases de datos del conjunto de datos de mezcla
    path_model_location = os.path.join(model_location, args.load_trainModel)
    path_model_location = os.path.join(path_model_location,'best.pt')
    model, optimizer, start_epoch, losslogger, accuracy, numEpochsNotImproving = load_checkpoint(model, optimizer, path_model_location)
    eval(
      args=args,
      model=model,
      optimizer=optimizer,
      device=device,
      model_location=model_location)
  
if __name__ == '__main__':
  main()