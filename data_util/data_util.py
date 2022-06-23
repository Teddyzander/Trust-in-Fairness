import numpy as np
import pandas as pd


def gen_data(size=1000000):
    """
    Generate the paper data set
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
        y = [-1, 1]
        target[row] = np.random.choice(y, p=[1/(1+np.exp(-2*a*y[0])), 1/(1+np.exp(-2*a*y[1]))])

        # allocate feature 2
        X_1 = data[row, 0]
        Y = target[row]
        data[row, 1] = X_1 + Y

        # allocate feature 3
        rand_1 = np.random.normal(a + 1, 1)
        rand_2 = np.random.normal(1 - 1, 1)
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
