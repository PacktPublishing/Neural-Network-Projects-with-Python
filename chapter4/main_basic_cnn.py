import matplotlib
matplotlib.use("TkAgg")
import os
import random
import warnings
warnings.filterwarnings("ignore")
from utils import train_test_split
from matplotlib import pyplot as plt

src = 'Dataset/PetImages/'

# Check if the dataset has been downloaded. If not, direct user to download the dataset first
if not os.path.isdir(src):
    print("""
          Dataset not found in your computer.
          Please follow the instructions in the link below to download the dataset:
          https://raw.githubusercontent.com/PacktPublishing/Neural-Network-Projects-with-Python/master/chapter4/how_to_download_the_dataset.txt
          """)
    quit()

# create the train/test folders if it does not exists already
if not os.path.isdir(src+'train/'):
    train_test_split(src)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define hyperparameters
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 128
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10

model = Sequential()
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

training_set = training_data_generator.flow_from_directory(src+'Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'binary')

test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = BATCH_SIZE,
                                             class_mode = 'binary')

model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=1)

score = model.evaluate_generator(test_set)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))



# Visualize results
test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size = (128, 128),
                                             batch_size = 1,
                                             class_mode = 'binary')

strongly_wrong_idx = []
strongly_right_idx = []
weakly_wrong_idx = []

for i in range(test_set.__len__()):
    img = test_set.__getitem__(i)[0]
    pred_prob = model.predict(img)[0][0]
    pred_label = int(pred_prob > 0.5)
    actual_label = int(test_set.__getitem__(i)[1][0])

    if pred_label != actual_label and (pred_prob > 0.9 or pred_prob < 0.1):
        strongly_wrong_idx.append(i)
    elif pred_label != actual_label and (pred_prob > 0.4 and pred_prob < 0.6):
        weakly_wrong_idx.append(i)
    elif pred_label == actual_label and (pred_prob > 0.9 or pred_prob < 0.1):
        strongly_right_idx.append(i)

def plot_on_grid(test_set, idx_to_plot, img_size=INPUT_SIZE):
    fig, ax = plt.subplots(3,3, figsize=(20,10))
    for i, idx in enumerate(random.sample(idx_to_plot,9)):
        img = test_set.__getitem__(idx)[0].reshape(img_size, img_size ,3)
        ax[int(i/3), i%3].imshow(img)
        ax[int(i/3), i%3].axis('off')
    plt.show()


plot_on_grid(test_set, strongly_right_idx)
plot_on_grid(test_set, strongly_wrong_idx)
plot_on_grid(test_set, weakly_wrong_idx)



