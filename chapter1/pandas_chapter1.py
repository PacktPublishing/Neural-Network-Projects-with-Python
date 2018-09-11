import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Pandas DataFrames
'''

# Read iris dataset from UCI database
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Get info of the data
print(df.info())
print('')

# Get statistical summary of the data
print(df.describe())
print('')

# Look at first 10 rows of the data
print(df.head(10))
print('')

# Select rows with sepal_length more than 5.0
df2 = df.loc[df['sepal_length'] > 5.0, ]

'''
Data Visualization in Panads
'''

# Define a color mapping by class
colors = df['class'].map({'Iris-setosa':'b', 'Iris-versicolor':'r', 'Iris-virginica':'g'})

# Then, plot a scatterplot using the color mapping
df.plot.scatter(x='sepal_length', y='sepal_width', color=colors, 
                title="Sepal Width vs Length by Species")
plt.show()
plt.clf()

# Plot histogram
df['petal_length'].plot.hist(title='Histogram of Petal Length')
plt.show()

# Plot boxplot
df.plot.box(title='Boxplot of Sepal Length & Width, and Petal Length & Width')
plt.show()

'''
Data Preprocessing in Pandas
'''

# Encode categorical variables
df2 = pd.DataFrame({'Day': ['Monday','Tuesday','Wednesday',
                           'Thursday','Friday','Saturday',
                           'Sunday']})
                           
# One-hot-encode
print(pd.get_dummies(df2))
print('')

# Imputing missing values
# Import the iris data once again
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Randomly select 10 rows
random_index = np.random.choice(df.index, replace= False, size=10)

# Set the sepal_length values of these rows to be None
df.loc[random_index,'sepal_length'] = None

# Check where the missing values are
print(df.isnull().any())
print('')

# Drop missing values
print("Number of rows before deleting: %d" % (df.shape[0]))
df2 = df.dropna()
print("Number of rows after deleting: %d" % (df2.shape[0]))
print('')

# Replace missing values with the mean
df.sepal_length = df.sepal_length.fillna(df.sepal_length.mean())

# Confirm that there are no missing values left
print(df.isnull().any())
print('')
