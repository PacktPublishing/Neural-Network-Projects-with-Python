import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
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
    fig, axes = plt.subplots(7, 5, figsize=(15,15))

    randomly_selected_imgs = random.sample(range(output_2_model.shape[0]),5)
    outputs = [X_test, output_1_model, output_2_model, output_4_model, output_8_model, output_16_model, output_32_model]

    # Iterate through each subplot and plot accordingly
    for row_num, row in enumerate(axes):
      for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][randomly_selected_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
