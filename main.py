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
parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
                    help='How many processes to load data')
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
parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
parser.add_argument('--eval', default=False, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to eval the model')
parser.add_argument('--version', type=str, default='v3',
                    help='Version to save the model')
parser.add_argument('--num-classes', type=int, default=12, metavar='N',
                    help='Number of training classes')
parser.add_argument('--load-epoch', type=int, default=-1,
                    help='Saved epoch to load and start training')


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



  # Model and xvectors path
  dirEmbeddings = 'cnn_rnn_' + args.version + '_classes_' + str(args.num_classes) + '_model_'
  if not args.is_googleColab:
    rootPath = os.getcwd()
    model_location = os.path.join(rootPath, 'models')
  else:
    model_location = os.path.join('/content/drive/My Drive', 'models')
  createDirectory(model_location)

  if args.train:
    if (args.load_epoch != -1):
      path_model_location = os.path.join(model_location, 'epoch-' + str(args.load_epoch) + '.pt')
      model, optimizer, start_epoch, losslogger, accuracy = load_checkpoint(model, optimizer, model_location)
    else:
      start_epoch = 0
      accuracy = 0
    train(
      args=args,
      model=model,
      start_epoch=start_epoch,
      accuracy=accuracy,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
      model_location=model_location)

if __name__ == '__main__':
  main()