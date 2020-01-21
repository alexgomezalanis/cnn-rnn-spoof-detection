from __future__ import print_function, division
import os
import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CNN_RNN_Dataset
from utils.collate import collate

rootPath = os.getcwd()

def test_epoch(model, device, data_loader, db_set, dirEmbeddings, dirSoftmax, db):
  with torch.no_grad():
    for batch_idx, sample in enumerate(data_loader):
      (x, labels, nameFiles) = sample
      embeddings, softmax = model(x)
      embeddings = embeddings.cpu().numpy()
      softmax = softmax.cpu().numpy()
      for n, nameFile in enumerate(nameFiles):
        np.save(os.path.join(dirEmbeddings, db, db_set, 'S' + str(labels[n]), nameFile + '.npy'), embeddings[n])
        np.save(os.path.join(dirSoftmax, db, db_set, 'S' + str(labels[n]), nameFile + '.npy'), softmax[n])

# db: LA or PA -> Embeddings being evaluated
# args.is_la: True / False -> Model is trained with LA
# db_set: training, development or test -> Dataset to evaluate
def eval(protocol, db, db_set, embeddings_location, softmax_location, args, model, device, mp):
  processes = []

  df = pd.read_csv('/home2/alexgomezalanis/tdnn-asvspoof-2019/spoof/protocols/' + protocol, sep=' ')
  numRows = len(df.index)
  rows_p = int(numRows / args.num_processes)

  for p in range(args.num_processes):
    if (p == args.num_processes - 1):
      df_p = df.iloc[p*rows_p:, :].reset_index().copy()
    else:
      df_p = df.iloc[p * rows_p : (p+1) * rows_p, :].reset_index().copy()
    dataset = CNN_RNN_Dataset(
      csv_file='',
      root_dir='/home2/alexgomezalanis',
      n_filts=args.num_filts,
      n_frames=args.num_frames,
      window=args.window_length,
      shift=args.frame_shift,
      dataset=db_set,
      is_evaluating_la=db == 'LA',    # Embeddings to evaluate
      dataframe=df_p,
      num_classes=args.num_classes)
  
    loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
      num_workers=args.num_data_workers, collate_fn=collate)

    process = mp.Process(target=test_epoch, args=(model, device, loader, db_set, embeddings_location, softmax_location, db))
    process.start()
    processes.append(process)
  
  for p in processes:
    p.join()