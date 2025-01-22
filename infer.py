

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image,ImageOps

if __name__ == '__main__':
  # print(tf.config.list_physical_devices('GPU'))
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path',help='Specify path to trained model',type=str)
  parser.add_argument('--data_folder',help='Specify folder holding data to infer on',type=str)
  parser.add_argument('--image_size',help='Specify side length of training/val/test images',type=int)

  args = parser.parse_args()



  ## perform same image processing steps
  image_paths = [os.path.join(args.data_folder,path) for path in os.listdir(args.data_folder)]
  images = np.array([Image.open(path).resize((args.image_size,args.image_size)) for path in image_paths])


  ## load model
  model = tf.keras.models.load_model(args.model_path)

  label_to_class = {0: 'adipose', 1: 'background', '2': 'debris', 3: 'lymphocytes', 4: 'mucus', 5: 'smooth muscle', 6: 'normal colon mucosa', 7: 'cancer-associated stroma', 8: 'colorectal adenocarcinoma epithelium'}
  ## perform inference on folder
  predictions = model.predict(images,verbose=0)
  results = pd.DataFrame({'image_path':image_paths,'Prediction':[label_to_class[np.argmax(x)] for x in predictions]})
  results.to_csv('results/predictions.csv')
  