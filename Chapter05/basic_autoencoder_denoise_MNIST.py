import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import random

def create_basic_autoencoder(hidden_layer_size):
  model = Sequential() 
  model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
  model.add(Dense(units=784, activation='sigmoid'))
  return model

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
  X_train_noised = X_train_reshaped + np.random.normal(0, 0.5, size=X_train_reshaped.shape) 
  X_test_noised = X_test_reshaped + np.random.normal(0, 0.5, size=X_test_reshaped.shape)
  X_train_noised = np.clip(X_train_noised, a_min=0, a_max=1)
  X_test_noised = np.clip(X_test_noised, a_min=0, a_max=1)

  # Model Building and Training
  basic_denoise_autoencoder = create_basic_autoencoder(hidden_layer_size=16)
  basic_denoise_autoencoder.compile(optimizer='adam', loss='mean_squared_error')
  basic_denoise_autoencoder.fit(X_train_noised, X_train_reshaped, epochs=10)

  output = basic_denoise_autoencoder.predict(X_test_noised)

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
    ax.imshow(X_test_noised[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
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
