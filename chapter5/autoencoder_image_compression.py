from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

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

    # Create autoencoders of different hidden layer size
    hiddenLayerSize_1_model = create_basic_autoencoder(hidden_layer_size=1)
    hiddenLayerSize_2_model = create_basic_autoencoder(hidden_layer_size=2)
    hiddenLayerSize_4_model = create_basic_autoencoder(hidden_layer_size=4)
    hiddenLayerSize_8_model = create_basic_autoencoder(hidden_layer_size=8)
    hiddenLayerSize_16_model = create_basic_autoencoder(hidden_layer_size=16)
    hiddenLayerSize_32_model = create_basic_autoencoder(hidden_layer_size=32)

    # Train each autoencoder
    hiddenLayerSize_1_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_1_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_2_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_2_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_4_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_4_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_8_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_8_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_16_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_16_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_32_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_32_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    # Use the trained models to make prediction on the testign set
    output_1_model = hiddenLayerSize_2_model.predict(X_test_reshaped)
    output_2_model = hiddenLayerSize_2_model.predict(X_test_reshaped)
    output_4_model = hiddenLayerSize_4_model.predict(X_test_reshaped)
    output_8_model = hiddenLayerSize_8_model.predict(X_test_reshaped)
    output_16_model = hiddenLayerSize_16_model.predict(X_test_reshaped)
    output_32_model = hiddenLayerSize_32_model.predict(X_test_reshaped)

    # Plot the output from each model to compare the results
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
          (ax11,ax12,ax13,ax14,ax15),(ax16,ax17,ax18,ax19,ax20),
          (ax21,ax22,ax23,ax24,ax25),(ax26,ax27,ax28,ax29,ax30),
          (ax31,ax32,ax33,ax34,ax35)) = plt.subplots(7, 5)

    randomly_selected_imgs = random.sample(range(output_1_model.shape[0]),5)

    # 1st row for original images
    for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
      ax.imshow(X_test[randomly_selected_imgs[i]], cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    # 2nd row for output images from autoencoder with hidden layer size = 1
    for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
      ax.imshow(output_1_model[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    # 3rd row for output images from autoencoder with hidden layer size = 2
    for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
      ax.imshow(output_2_model[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    # 4th row for output images from autoencoder with hidden layer size = 4
    for i, ax in enumerate([ax16,ax17,ax18,ax19,ax20]):
      ax.imshow(output_4_model[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    # 5th row for output images from autoencoder with hidden layer size = 8
    for i, ax in enumerate([ax21,ax22,ax23,ax24,ax25]):
      ax.imshow(output_8_model[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    # 6th row for output images from autoencoder with hidden layer size = 16
    for i, ax in enumerate([ax26,ax27,ax28,ax29,ax30]):
      ax.imshow(output_16_model[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    # 7th row for output images from autoencoder with hidden layer size = 32
    for i, ax in enumerate([ax31,ax32,ax33,ax34,ax35]):
      ax.imshow(output_32_model[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
      ax.grid(False)
      ax.set_xticks([])
      ax.set_yticks([])

    plt.show()
