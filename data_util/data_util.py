import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def gen_data(size=1000000, t=0):
    """
    Generate the paper data set
    :param size: the size of the generated data set
    :param t: t is a value between [0,1], and is a linear transformation that represents the bias in the data. t=0 is
    maximum bias, and t=1 is no bias.
    :return: the data set
    """

    # seed the pseudo-random number generator so experiments can be repeated
    np.random.seed(123)

    # preallocate space for data
    data = np.asarray([np.random.normal(loc=0.0, scale=1.0, size=size),
                       np.random.normal(loc=0.0, scale=1.0, size=size),
                       np.random.normal(loc=0.0, scale=1.0, size=size),
                       np.random.choice([-1, 1], size=size)])
    data = np.transpose(data)

    target = np.transpose(np.array(np.random.choice([-1, 1], size=size)))

    # generate the causality model
    for row in range(0, size):
        # allocate target
        a = data[row, 3]
        if a == -1:
            a = t * (-2 * a) - 1
        y = [-1, 1]
        target[row] = np.random.choice(y, p=[1/(1+np.exp(-2*a*y[0])), 1/(1+np.exp(-2*a*y[1]))])

        # allocate feature 2
        X_1 = data[row, 0]
        Y = target[row]
        data[row, 1] = X_1 + Y

        # allocate feature 3
        rand_1 = np.random.normal(a + 1, 1)
        rand_2 = np.random.normal(a - 1, 1)
        data[row, 2] = (1/(1+np.exp(-2*a))) * rand_1 + (1/(1+np.exp(2*a))) * rand_2

    # convert to a data frame
    data = pd.DataFrame(data,
                        columns=['A', 'B', 'C', 'D'])

    target = pd.Series(target)

    # make the protected feature discrete
    for col in ['D']:
        data[col] = data[col].astype('category')

    # save protected feature column name
    sens = 'D'

    return data, target, sens


def get_data_type(data):
    """
    Takes in a data frame with headers and returns the data type (continuous [d] or discrete [c]) of each feature
    :param data: pandas data frame
    :return: Three lists. First list contains discrete labels, second list contains continuous labels, third list
    contains the order of label types eg ['c', 'd', 'd', ...]
    """

    # get number of labels and allocate memory to store list of label types
    labels = data.columns
    num_labels = len(np.asarray(labels))
    cat = [str] * num_labels

    # split labels into continuous list and discrete list
    cont_labels = np.asarray(data.select_dtypes('number').columns)
    dis_labels = np.asarray(data.select_dtypes('category').columns)
    dis_labels = np.concatenate((dis_labels, np.asarray(data.select_dtypes('object').columns)))

    # create list that defines all columns as either continuous 'c' or discrete 'd'
    for i in range(0, num_labels):
        if labels[i] in cont_labels:
            cat[i] = 'c'
        else:
            cat[i] = 'd'

    return dis_labels, cont_labels, cat


def normalise(data):
    """
    Takes a data column and makes data->[0,1]
    :param data: input data
    :return: normalised data
    """
    data_max = np.max(data)
    data_min = np.min(data)
    if data_max == data_min:
        data = 0
    else:
        data = (data - data_min) / (data_max - data_min)

    return data


def normalise_data(data, dis_labels, con_labels):
    """
    standardises data such that discrete data is numeric and continuous data has mean 0 and variance 1
    :param data: original dataframe
    :param dis_labels: list of discrete labels
    :param con_labels: list of continuous labels
    :return: standardises data set
    """

    # standardise discrete labels such that the feature is numeric
    for label in dis_labels:
        data[label] = data[label].astype('category').cat.codes

    # normalise the continuous data
    for label in con_labels:
        data[label] = normalise(data[label])

    return data


def split(data, target, sensitive, ratio=0.7, seed=876):
    """
    Splits the data into training data and testing data
    :param data: input data
    :param target: target data
    :param sensitive: sensitive data
    :param ratio: ratio of data split (eg 0.7 is 70% training, 30% testing)
    :param seed: set pseudo-random seed so experiments can be repeated with same test/train split
    :param sens_name: name of sensitive label
    :return: x_tr is the training input, y_tr is the testing output, sens_tr is the training sensitive data input,
    x_te is the testing input, y_te is the testing output, sens_te is the testing sensitive data input
    """

    x_tr, x_te, y_tr, y_te, sens_tr, sens_te = train_test_split(data, target, sensitive, train_size=ratio, random_state=seed)
    return x_tr, y_tr, sens_tr, x_te, y_te, sens_te
