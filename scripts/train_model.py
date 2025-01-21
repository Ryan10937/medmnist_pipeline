import numpy as np
import tensorflow as tf
def train_model(model,dataset_dict:dict,data_limit,epochs):
  train_len = len(dataset_dict['train_dataset']) if data_limit is None else data_limit
  # train_history_arr = []
  # for epoch in range(epochs):
  #   train_counter = 0
  #   for (train_x,train_y),(val_x,val_y) in zip(dataset_dict['train_loader'],dataset_dict['validation_loader']):
  #     train_counter+=1
  #     print(f'{epoch} {train_counter*len(train_x)}/{train_len}')
  #     training_history = model.fit(train_x,train_y,batch_size=len(train_x),validation_data=(val_x,val_y),verbose=0,epochs=1)
  #     train_history_arr.append(training_history.history)
  #     if train_counter*len(train_x) >= train_len:
  #       break
  X=np.array([x for x,y in dataset_dict['train_dataset']])
  y=np.array([y for x,y in dataset_dict['train_dataset']])
  val_X=np.array([x for x,y in dataset_dict['validation_dataset']])
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