import data_util.data_util as data_util


if __name__ == '__main__':

    # get the causality model data
    data, target, sens = data_util.gen_data()

    # get column labels
    dis_labels, con_labels, cat = data_util.get_data_type(data)

    # normalise the data
    data = data_util.normalise_data(data, dis_labels, con_labels)

    # normalise the target
    target = target.astype('category').cat.codes

    print('stop')

