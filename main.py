import data_util.data_util as data_util


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # get the causality model data
    data, target, sens = data_util.gen_data()

    # get column labels
    dis_labels, cont_labels, cat = data_util.get_data_type(data)

    print('stop')

