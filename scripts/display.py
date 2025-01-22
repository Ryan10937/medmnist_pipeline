import matplotlib.pyplot as plt
import numpy as np

def create_visualizations(information:dict):

  #create training history plots
  train_history_plots(information)
  

  #create dataset information plots
  get_dataset_information(information)

def get_dataset_information(information:dict):
  classes,counts = information['class_counts']
  plt.figure()
  plt.bar(classes,height=counts)
  plt.title(f'Class Distribution')
  plt.xlabel('Class')
  plt.ylabel('Count')
  plt.savefig('figures/class_distribution.png')

def train_history_plots(information):
  train_history_arr = information['train_history_arr'].history
  plt.figure()
  plt.plot(np.array([hist['accuracy'] for hist in train_history_arr]).flatten())
  plt.plot(np.array([hist['val_accuracy'] for hist in train_history_arr]).flatten())
  plt.title('Training History')
  plt.legend(['Train Accuracy','Validation Accuracy'])
  plt.savefig('figures/train_accuracy_history.png')

  plt.figure()
  plt.title('Training History')
  plt.plot(np.array([hist['loss'] for hist in train_history_arr]).flatten())
  plt.plot(np.array([hist['val_loss'] for hist in train_history_arr]).flatten())
  plt.legend(['Train Loss','Validation Loss'])
  plt.savefig('figures/train_loss_history.png')