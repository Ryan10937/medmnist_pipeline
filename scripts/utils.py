
from medmnist import INFO
import scripts.dataset_without_pytorch as dataset_without_pytorch
from scripts.dataset_without_pytorch import get_loader
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

def get_dataset(data_flag,batch_size,image_size):
  # data_flag = 'pathmnist'
  # data_flag = 'breastmnist'
  download = True

  BATCH_SIZE = batch_size

  info = INFO[data_flag]
  task = info['task']
  n_channels = info['n_channels']
  n_classes = len(info['label'])

  DataClass = getattr(dataset_without_pytorch, info['python_class'])
  # load the data
  train_dataset = DataClass(split='train', download=download,size=image_size)
  validation_dataset = DataClass(split='val', download=download,size=image_size)
  test_dataset = DataClass(split='test', download=download,size=image_size)
  
  # encapsulate data into dataloader form
  train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
  validation_loader = get_loader(dataset=validation_dataset, batch_size=BATCH_SIZE)
  test_loader = get_loader(dataset=test_dataset, batch_size=BATCH_SIZE)
  return {
    'train_dataset':train_dataset,
    'validation_dataset':validation_dataset,
    'test_dataset':test_dataset,
    'train_loader':train_loader,
    'validation_loader':validation_loader,
    'test_loader':test_loader,
    }

def load_model(image_size):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
    base_model.trainable = False
    model=tf.keras.Sequential([
      # tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(image_size,image_size,3)),#may need to be greyscale
      # tf.keras.layers.MaxPooling2D((2, 2)),
      # tf.keras.layers.Conv2D(128,(3,3),activation='relu'),#may need to be greyscale
      # tf.keras.layers.MaxPooling2D((2, 2)),
      # tf.keras.layers.Flatten(),
      # tf.keras.layers.Dense(128,activation='sigmoid'),
      # tf.keras.layers.Dense(128,activation='sigmoid'),
      # tf.keras.layers.Dense(128,activation='sigmoid'),
      # tf.keras.layers.Dense(128,activation='sigmoid'),

      base_model,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(9,activation='softmax'),
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model