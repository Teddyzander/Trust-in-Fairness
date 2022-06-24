import data_util.data_util as data_util
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import demographic_parity_difference

if __name__ == '__main__':
    # get the causality model data
    data, target, sens = data_util.gen_data()

    # get column labels
    dis_labels, con_labels, cat = data_util.get_data_type(data)

    # normalise the data
    data = data_util.normalise_data(data, dis_labels, con_labels)

    # normalise the target
    target = target.astype('category').cat.codes

    # save sensitive column, but remove it from the data
    sensitive = data['D']
    data = data.drop('D', axis=1)

    # split the data into training and testing sets
    x_tr, y_tr, sens_tr, x_te, y_te, sens_te = data_util.split(data, target, sensitive)

    # fit model to training data
    model = LogisticRegression(max_iter=10000)
    model.fit(x_tr, y_tr)

    print(model.score(x_te, y_te))

    baseline_output = model.predict(x_te)
    base_fairness = demographic_parity_difference(y_te, baseline_output, sensitive_features=sens_te)

    print(base_fairness)

    print('stop')
