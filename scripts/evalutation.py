import sklearn.metrics as metrics
import numpy as np
def evaluate_model(model,dataset_dict:dict):
  test_len = len(dataset_dict['test_dataset'])
  test_counter=0
  predictions = []
  labels = []
  for test_x,test_y in dataset_dict['test_loader']:
    test_counter+=1
    predictions.append(model.predict(test_x,verbose=0))
    labels.append(test_y)
    if test_counter*len(test_x) >= test_len:
      break

  acc = metrics.accuracy_score([y for x in labels for y in x],[np.argmax(y) for x in predictions for y in x])
  print('Accuracy:',acc)