from scripts.utils import get_dataset,load_model
from scripts.display import create_visualizations
from scripts.evalutation import evaluate_model
from scripts.train_model import train_model

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
  print(tf.config.list_physical_devices('GPU'))
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs',help='Specify number of model training epochs',type=int)
  parser.add_argument('--image_size',help='Specify side length of training/val/test images',type=int)
  parser.add_argument('--batch_size',help='Specify batch size used during training',type=int)
  parser.add_argument('--data_limit',help='Specify limit for dataset. Used in debugging only',required=False,type=int)
  parser.add_argument('--dataset_name',help='Specify medmnist dataset name',type=str)
  args = parser.parse_args()
  
  plotting_info={}


  dataset_dict = get_dataset(args.dataset_name,args.batch_size,image_size=args.image_size)#get dataset from medmnist
  


  #train model, store training history for later
  model = load_model(args.image_size)#define and compile model
  train_history_arr = train_model(model,dataset_dict,args.data_limit,args.epochs)


  # Gather information for figure generation
  plotting_info['class_counts'] = np.unique(np.array([y for x,y in dataset_dict['train_dataset']]),return_counts=True)
  plotting_info['train_history_arr']=train_history_arr
  create_visualizations(plotting_info)

  #get model metrics and print
  evaluate_model(model,dataset_dict=dataset_dict)