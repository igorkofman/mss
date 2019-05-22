import tensorflow as tf
import musdb
import numpy as np
#tf.enable_eager_execution()

db = musdb.DB('/Users/igor/src/ml/mss/data/musdb18')
track = db.load_mus_tracks()[0]

# cast to float32 and tensor
x = tf.transpose(tf.convert_to_tensor(track.audio.astype(np.float32)))[0]
y = tf.transpose(tf.convert_to_tensor(track.stems[0].astype(np.float32)))[0]

# do STFT
frame_length = 1024
frame_step = 512
fft_length = 1024
x_stfts = tf.signal.stft(x, 
    frame_length=frame_length, 
    frame_step=frame_step,
    fft_length=fft_length)
y_stfts = tf.signal.stft(y, 
    frame_length=frame_length, 
    frame_step=frame_step,
    fft_length=fft_length)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
samples = int(track.duration * track.rate / frame_step) - 1
num_layers = 3
layer_size = 512 

#model.add(Flatten(input_shape=(frame_step+1,)))
model.add(Dense(layer_size, input_shape=(frame_step+1,), activation='relu'))


for _ in range(num_layers-2):
    model.add(Dense(layer_size, activation='relu'))
#    model.add(Dropout(dropout_amount))
model.add(Dense(frame_step + 1, activation='softmax'))
# Your code above (Lab 1)
model.compile('adam', 'kullback_leibler_divergence') 


model.fit(x_stfts, y_stfts, steps_per_epoch=1000, epochs=10,  verbose = 1)

#x_stfts
#y_pred_stfts = model.predict(stfts)
print (model.evaluate(x_stfts, y_stfts, steps=16))
