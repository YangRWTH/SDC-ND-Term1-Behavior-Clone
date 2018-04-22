# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:47:40 2018

@author: Yang Liu
"""

#%% 



import help_functions as hf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential

#%% Build the model 

def build_model(keep_prob):

#   Modified NVIDIA model
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0,input_shape = INPUT_SHAPE ))
    model.add(Conv2D(24, 5, 5, activation='relu'))
    model.add(Conv2D(36, 5, 5, activation='relu'))
    model.add(Conv2D(48, 5, 5, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))    
    return model


#%% Define parameters
EPOCHS = 2
BATCH_SIZE = 128 
INPUT_SHAPE = (66, 200, 3)
KEEP_PROB = 0.5
csv_files  = 'data/driving_log.csv'
image_path = 'data/IMG'


#%% get trainin data in form of a list
train_sets = hf.get_training_set(csv_files, image_path)
print('length of the training set'.format(len(train_sets)))


#%% generate training data and validation data

train_samples, validation_samples = train_test_split(train_sets, test_size = 0.2)
train_generator = hf.generator(image_path, train_sets, batch_size = BATCH_SIZE)
validation_generator = hf.generator(image_path, validation_samples, batch_size=BATCH_SIZE)

#%% Compile the model und do the traning
model = build_model(KEEP_PROB)

model.compile(loss='mse',
              optimizer='adam')


history_obj = model.fit_generator(train_generator, 
                    samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples),
                    verbose=1,
                    nb_epoch= EPOCHS)

model.save('model.h5')

#%%
### print the keys contained in the history object
print(history_obj.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()