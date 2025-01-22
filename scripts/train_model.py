import numpy as np
import tensorflow as tf
from scripts.image_augmentation import augment_images
def train_model(model,dataset_dict:dict,data_limit,epochs):
  
  X=np.array(augment_images([x for x,y in dataset_dict['train_dataset']]))
  y=np.array([y for x,y in dataset_dict['train_dataset']])
  val_X=np.array(augment_images([x for x,y in dataset_dict['validation_dataset']]))
  val_y=np.array([y for x,y in dataset_dict['validation_dataset']])
  train_history = model.fit(X,
                            y,
                            epochs=epochs,
                            batch_size=64,
                            validation_data = (val_X,val_y),
                            callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                       patience=10,
                                                                       restore_best_weights=True)
                            )
  model.save('models/model.keras')
  return train_history