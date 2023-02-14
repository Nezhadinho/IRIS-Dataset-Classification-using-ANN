# IRIS-Dataset-Classification-using-ANN
Classify the IRIS Dataset using ANN from scratch.

### Introduction

The IRIS dataset contains 150 samples in 3 different classes named "Setosa", "Versicolor", and "Virginica". There are 4 factors which determine the class of each sample. The Factors are sepal length in cm, sepal width in cm, petal length in cm, and petal width in cm. For more Information and to download the dataset you can simply go to this [LINK](https://archive.ics.uci.edu/ml/datasets/iris).

### Importing libraries

Obviously, the first part of each project is to import necessary libraries and since this project is from scratch we only import 3 libraries.

```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

### Data preprocessing

To load the Data, you can download it from the link in the [Introduction](https://github.com/Nezhadinho/IRIS-Dataset-Classification-using-ANN/edit/main/README.md#introduction), then you can read the data using pandas library.

```
data = pd.read_csv("[file directory]")
```

To get a simple information about the data you can use ```data.head()``` and see what the data looks like.

```
print(data.head())
```
The output should be something like this.

| |sepal.length|sepal.width|petal.length|petal.width|variety|
|-|------------|-----------|------------|-----------|-------|
|0|5.1|3.5|1.4|0.2|Setosa|
|1|4.9|3.0|1.4|0.2|Setosa|
|2|4.7|3.2|1.3|0.2|Setosa|
|3|4.6|3.1|1.5|0.2|Setosa|
|4|5.0|3.6|1.4|0.2|Setosa|

To work with the data you should convert it to array. To do that you can use numpy library.

```
data = np.array(data)
```

In order to have a fair training set, use ```random.shuffle()```.
```
np.random.shuffle(data)
```
Note that ```data = np.random.shuffle(data)``` does not work.

At this point, we are ready to start splitting the data into training and test set. It is optional but in general, putting 80% of the data in the training data set is the optimum ratio. In this case, we have 120 samples in training set and 30 samples in the testing set. Do not forget to ```reshape()``` them to avoid furthere errors.
```
data_train = data[:-30].T
x_train = data_train[:-1]
x_train = x_train.reshape(120, 4)
y_train = data_train[-1]
y_train = y_train.reshape(120, 1)

data_test = data[-30:].T
x_test = data_test[:-1]
x_test = x_test.reshape(30, 4)
y_test = data_test[-1]
y_test = y_test.reshape(30, 1)
```

### Building ANN

##### Intro
This section has 3 main parts. The first part is feeding the input data to the network and then calculate the output. The second part is calculating the backward propagation in order calculate the error with respect to each weight. Last but not least, updating the weights.

In this project I am using the simplest ANN which has 2 layers, one hidden layer with 10 neurons and one output layer.
![Untitled](https://user-images.githubusercontent.com/82404535/218729586-2c2279a5-bfbd-4589-8b50-c644e4ce03ea.jpg)
