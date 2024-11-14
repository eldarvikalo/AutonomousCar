import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from functions import load_img_steering, batch_generator, nvidia_model

# Data Preparation
datadir = 'Track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', None)
data['center'] = data['center'].apply(lambda x: x.split('/')[-1])
data['left'] = data['left'].apply(lambda x: x.split('/')[-1])
data['right'] = data['right'].apply(lambda x: x.split('/')[-1])

# Balance the steering angles (same as in original code)
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
print('total data:', len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))

# Load image paths and steering angles
image_paths, steerings = load_img_steering(datadir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

# Training and Validation Batch Generators
train_generator = batch_generator(X_train, y_train, 100, istraining=True)
valid_generator = batch_generator(X_valid, y_valid, 100, istraining=False)

# Model
model = nvidia_model()
print(model.summary())

# Train the model
history = model.fit_generator(train_generator,
                    steps_per_epoch=300,
                    epochs=10,
                    validation_data=valid_generator,
                    validation_steps=200,
                    verbose=1,
                    shuffle=1)

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

# Save model
model.save('./Model/model.h5')
