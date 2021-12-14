import numpy as np

import tensorflow as tf

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import matplotlib as plt

training_set_data_list = []
training_set_class_list = []
training_set_onehot_list = []


validation_set_data_list = []
validation_set_class_list = []
validation_set_onehot_list = []

test_set_data_list = []
test_set_data_list_unsplitted = []

for i in range(1,858) :
    a = np.load('3mc/training_set/'+str(i)+'.npy',allow_pickle=True)
    j = 0
    lim = a[0].shape[0]/96
    a[2] = np.array(a[2])
    while j < lim :

        training_set_data_list.append(a[0][(j*96) : ((j+1)*96) ][:])
        training_set_class_list.append(a[1])
        training_set_onehot_list.append(a[2])
        j += 1

for i in range(858,1028) :

    a = np.load('3mc/validation_set/'+str(i)+'.npy',allow_pickle=True)
    j = 0
    lim = a[0].shape[0]/96
    a[2] = np.array(a[2])
    while j < lim :

        validation_set_data_list.append(a[0][(j*96) : ((j+1)*96) ][:])
        validation_set_class_list.append(a[1])
        validation_set_onehot_list.append(a[2])
        j += 1

for i in range(1,171) :

    a = np.load('3mc/test_set/'+str(i)+'.npy',allow_pickle=True)
    print(a)
    print(a.shape)
    test_set_data_list_unsplitted.append(a)
    j = 0
    lim = a.shape[0]/96
    while j < lim :

        test_set_data_list.append(a[(j*96) : ((j+1)*96) ][:])
        j += 1

#training_set_data_list = training_set_data_list.reshape(-1, 96,64,1)
for t in training_set_data_list :
    t = t.reshape(-1,96,64,1)

for v in validation_set_data_list :
    v = v.reshape(-1,96,64,1)

for k in test_set_data_list :
    k = k.reshape(-1,96,64,1)


batch_size = 64
epochs = 10
num_classes = 3

fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=(96,64,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2,2), padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3,3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Conv2D(128, (3,3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128,activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

fashion_model.summary()

train_fit = np.array(training_set_data_list)
train_fit_onehot = np.array(training_set_onehot_list)
validation_fit = np.array(validation_set_data_list)
validation_fit_onehot = np.array(validation_set_onehot_list)
test_set = np.array(test_set_data_list)

fashion_train = fashion_model.fit( train_fit, train_fit_onehot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(validation_fit,validation_fit_onehot))

predicted_classes = fashion_model.predict(test_set)

print(predicted_classes)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(predicted_classes)
predictions = []

count = 1
j = 0

vect = np.array([0,0,0])

submit = open("submission1.csv","w")
submit.write("ID,Class\n")
for i in range(0,170) :

    seconds = test_set_data_list_unsplitted[i].shape[0] / 96
    print('Seconds : ',seconds)
    pred = predicted_classes[j:int(seconds+j)]
    
    for k in pred :
        vect[k] += 1

    predictions.append(np.argmax(vect))
    print(str(count)+'.npy',predictions[i])
    submit.write(str(count)+'.npy,'+str(predictions[i])+'\n')
    count += 1
    vect[:] = 0
    j += int(seconds)

print(predictions)
print('--------------------')

count = 1
for k in predicted_classes :
    print(count,k)
    count += 1
