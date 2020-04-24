import torch
import os

def load_checkpoint(model, optimizer, filename):
  # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
  start_epoch = 0
  optimizer=optimizer
  model = model
  losslogger=0
  accuracy=0
  numEpochsNotImproving=0
  if os.path.isfile(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    losslogger = checkpoint['losslogger']
    accuracy = checkpoint['accuracy']
    numEpochsNotImproving = checkpoint['numEpochsNotImproving']
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
  else:
    print("=> no checkpoint found at '{}'".format(filename))

  return model, optimizer, start_epoch, losslogger, accuracy, numEpochsNotImproving