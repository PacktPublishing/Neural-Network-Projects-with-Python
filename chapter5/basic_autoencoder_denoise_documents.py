import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from matplotlib import pyplot as plt
import numpy as np
import os
import random
from keras.preprocessing.image import load_img, img_to_array

if __name__ == '__main__':
    # Import noisy office documents dataset
    noisy_imgs_path = 'Noisy_Documents/noisy/'
    clean_imgs_path = 'Noisy_Documents/clean/'

    X_train_noisy = []
    X_train_clean = []

    for file in sorted(os.listdir(noisy_imgs_path)):
      img = load_img(noisy_imgs_path+file, color_mode='grayscale', target_size=(420,540))
      img = img_to_array(img).astype('float32')/255
      X_train_noisy.append(img)

    for file in sorted(os.listdir(clean_imgs_path)):
      img = load_img(clean_imgs_path+file, color_mode='grayscale', target_size=(420,540))
      img = img_to_array(img).astype('float32')/255
      X_train_clean.append(img) 

    # convert to numpy array
    X_train_noisy = np.array(X_train_noisy) 
    X_train_clean = np.array(X_train_clean)

    # use the first 20 noisy images as testing images
    X_test_noisy = X_train_noisy[0:20,]
    X_train_noisy = X_train_noisy[21:,]

    # use the first 20 clean images as testing images
    X_test_clean = X_train_clean[0:20,]
    X_train_clean = X_train_clean[21:,]

    # Build and train model
    basic_conv_autoencoder = Sequential()
    basic_conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same', input_shape=(420,540,1)))
    basic_conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
    basic_conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))
    basic_conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    basic_conv_autoencoder.fit(X_train_noisy, X_train_clean, epochs=10)

    output = basic_conv_autoencoder.predict(X_test_noisy)

    # Plot Output
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)

    randomly_selected_imgs = random.sample(range(X_test_noisy.shape[0]),2)

    for i, ax in enumerate([ax1, ax4]):
        idx = randomly_selected_imgs[i]
        ax.imshow(X_test_noisy[idx].reshape(420,540), cmap='gray')
        if i == 0:
            ax.set_title("Noisy Images")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate([ax2, ax5]):
        idx = randomly_selected_imgs[i]
        ax.imshow(X_test_clean[idx].reshape(420,540), cmap='gray')
        if i == 0:
            ax.set_title("Clean Images")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate([ax3, ax6]):
        idx = randomly_selected_imgs[i]
        ax.imshow(output[idx].reshape(420,540), cmap='gray')
        if i == 0:
            ax.set_title("Output Denoised Images")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()