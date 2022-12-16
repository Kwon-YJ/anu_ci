import os
import io
import random
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping
import cv2
from sklearn.model_selection import train_test_split
import mtcnn
from keras.callbacks import ModelCheckpoint
from PIL import Image
import time
import sys

import matplotlib.image as img
import matplotlib.pyplot as pp


BATCH_SIZE = 16
EPOCH = 1000
face_data = "/home/declan/yeongseo/Face_Recognition/doctor/dataset/resize_dataset/train/"
num_classes = len(os.listdir(face_data))
print('num classes = ', num_classes)

X = []
Y = []
X2 = []
Y2 = []


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors

    return K.sqrt(K.maximum(sumSquared, K.epsilon()))



def make_pairs(x, y):

    #print('y: ', y)


    pairs = []
    labels = []

    # add a matching example
    #for iter in range(0, (num_classes/2)):


    for i in range(0, len(y)):
        for j in range(0, len(y)):
            # add a matching example
            if y[i] == y[j]:
                pairs += [[x[i], x[j]]]
                labels += [0]

            # add a non-matching example
            elif y[i] != y[j]:
                pairs += [[x[i], x[j]]]
                labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()



class lossStop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, eopch, logs={}):
        if (logs.get('loss') < 0.0005):
            #print("\n----reach 60% accuracy, stop training----")
            self.model.stop_training = True


lossStop = lossStop()


for idex, face_names in enumerate(os.listdir(face_data)):
    label = [0 for i in range(num_classes)]
    label =  idex
    print('label: ', label)
    person_dir = os.path.join(face_data,face_names)
    i = 0
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)
        #img = Image.open(image_path)
        img = cv2.imread(image_path)
        print(image_path+'\n')
        
        if(len(img) != 0):
            print('image add')
            img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img/255)
            Y.append(label)
            print(label)
            del img       
        
            X.append(k/255)
            Y.append(label)
            print(label)
            del k

X = np.array(X).astype('float32')
Y = np.array(Y)



print("X: ", Y.shape)
print("Y: ", Y.shape)



pairs_train, labels_train = make_pairs(X, Y)
del X
del Y


idx = np.arange(pairs_train.shape[0])
np.random.shuffle(idx)


pairs_train = pairs_train[idx]
labels_train = labels_train[idx]



x_train1 = pairs_train[:, 0]
x_train2 = pairs_train[:, 1]
print("x_train1: ", x_train1.shape)
print("x_train2: ", x_train2.shape)

visualize(pairs_train[:-1] labels_train[:-1], to_show=101, num_col=10)
visualize(pairs_test[:-1], labels_test[:-1], to_show=51, num_col=10)



input = tf.keras.layers.Input(shape = (112, 112, 3))
x = tf.keras.applications.ResNet152V2(weights = 'imagenet', include_top=False, input_shape = (112, 112, 3))(input)
x.trainable=False
x = tf.keras.layers.Dense(1024)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(10)(x)

embedding_model = tf.keras.models.Model(inputs = input, outputs = x)

embedding_model.summary()

input_a = tf.keras.layers.Input(shape = (112, 112, 3))
input_b = tf.keras.layers.Input(shape = (112, 112, 3))

model_a = embedding_model(input_a)
model_b = embedding_model(input_b)


distance = tf.keras.layers.Lambda(euclidean_distance)([model_a, model_b])
#normal_layer = tf.keras.layers.BatchNormalization()(distance)
output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
facemodel = tf.keras.models.Model(inputs = [input_a, input_b], outputs = output)
 
facemodel.compile(tf.keras.optimizers.Adam(learning_rate = 0.001), loss=loss(margin=1), metrics=['accuracy'])

facemodel.summary()

checkpoint_path = "training_5ours/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath='siamese_resnet152_221104_best_ours.h5',
                                             monitor='loss',
                                             save_best_only=True),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True),
             lossStop]


#history = face_model.fit(x = [x_train[:36], x_train[36:]], y= [y_train[:36], y_train[36:]], epochs = EPOCH)
#history = facemodel.fit([x_train1, x_train2], labels_train, epochs = EPOCH, batch_size = BATCH_SIZE, validation_data = ([x_test1, x_test2], labels_test), callbacks = callbacks)
history = facemodel.fit([x_train1, x_train2], labels_train, epochs = EPOCH, batch_size = BATCH_SIZE, callbacks = callbacks)
    

facemodel.save('siamese_resnet152_221104_ours.h5')
facemodel.save_weights('siamese_resnet152_221104_weight_ours.h5')

#정확도 시각화
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy & Loss')
plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')
fig2 = plt.gcf()
fig2.savefig('siamese_resnet152_221104_ours.png')


ha = cv2.imread('ours_data/Yeongseo_Ha/0.jpg')
park = cv2.imread('ours_data/Jihee_Park/0.JPG')
shim = cv2.imread('ours_data/Jaechang_Shim/0.JPG')


ha = cv2.cvtColor(ha, cv2.COLOR_BGR2RGB)
park = cv2.cvtColor(park, cv2.COLOR_BGR2RGB)
shim = cv2.cvtColor(shim, cv2.COLOR_BGR2RGB)

ha = cv2.resize(ha, (112, 112))
ha = np.array(ha/255)
ha = np.expand_dims(ha, axis = 0)

park = cv2.resize(park, (112, 112))
park = np.array(park/255)
park = np.expand_dims(park, axis = 0)

shim = cv2.resize(shim, (112, 112))
shim = np.array(shim/255)
shim = np.expand_dims(shim, axis = 0)

predictions = facemodel.predict([ha, park])
print('ha vs park: ', predictions[0][0])
del predictions

predictions = facemodel.predict([ha, shim])
print('ha vs shim: ', predictions[0][0])
del predictions

predictions = facemodel.predict([park, shim])
print('park vs shim: ', predictions[0][0])
del predictions

