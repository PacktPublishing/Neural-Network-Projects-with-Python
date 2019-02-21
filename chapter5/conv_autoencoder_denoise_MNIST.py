import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from matplotlib import pyplot as plt
import numpy as np
import random

if __name__ == '__main__':
  # Import MNIST dataset
  training_set, testing_set = mnist.load_data()
  X_train, y_train = training_set
  X_test, y_test = testing_set

  # Reshape the dataset for our neural network
  X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
  X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

  # Normalize range of values between 0 to 1 (from 0 to 255)
  X_train_reshaped = X_train_reshaped/255.
  X_test_reshaped = X_test_reshaped/255.

  # Add noise to the MNIST dataset
  X_train_noisy = X_train_reshaped + np.random.normal(0, 0.5, size=X_train_reshaped.shape) 
  X_test_noisy = X_test_reshaped + np.random.normal(0, 0.5, size=X_test_reshaped.shape)
  X_train_noisy = np.clip(X_train_noisy, a_min=0, a_max=1)
  X_test_noisy = np.clip(X_test_noisy, a_min=0, a_max=1)

  # Model Building and Training
  conv_autoencoder = Sequential()
  conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28,28,1)))
  conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
  conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
  conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
  conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))
  conv_autoencoder.summary()
  conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  conv_autoencoder.fit(X_train_noisy.reshape(60000,28,28,1), X_train_reshaped.reshape(60000,28,28,1), epochs=10)

  output = conv_autoencoder.predict(X_test_noisy.reshape(10000,28,28,1))

  # Plot output
  fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5)
  randomly_selected_imgs = random.sample(range(output.shape[0]),5)

  # 1st row for original images
  for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(X_test_reshaped[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
    if i == 0:
      ax.set_ylabel("Original \n Images")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

  # 2nd row for input with noise added
  for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(X_test_noisy[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
    if i == 0:
      ax.set_ylabel("Input With \n Noise Added")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

  # 3rd row for output images from our autoencoder
  for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
    if i == 0:
      ax.set_ylabel("Denoised \n Output")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

  plt.show()


