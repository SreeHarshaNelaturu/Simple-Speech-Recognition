from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

feature_dim2 = 11

save_data_to_array(max_len = feature_dim2)

X_train, X_test, y_train, y_test = get_train_test()

feature_dim1 = 20
channel = 1
epochs = 15
batch_size = 100
verbose = 1
num_classes = 3

X_train = X_train.reshape(X_train.shape[0], feature_dim1, feature_dim2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim1, feature_dim2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def speech_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation="relu", input_shape = (feature_dim1, feature_dim2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation="relu"))
    model.add(Conv2D(64, kernel_size=(2,2), activation="relu"))
    model.add(Conv2D(120, kernel_size=(2,2), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001, decay=0), metrics=["accuracy"])
    return model


def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim1, feature_dim2, channel)
    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]

model = speech_model()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

print(predict(<Path of test wav file>, model=model))
