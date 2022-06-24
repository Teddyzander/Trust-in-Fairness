import data_util.data_util as data_util
import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer

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

    # fit threshold optimiser to model
    thresh_model = ThresholdOptimizer(
            estimator=model,
            constraints='demographic_parity',
            objective='accuracy_score',
            prefit=True,
            grid_size=100000)
    thresh_model.fit(x_tr, y_tr, sensitive_features=sens_tr)

    # make predictions on testing data set
    baseline_output = model.predict(x_te)
    thresh_output = thresh_model.predict(x_te, sensitive_features=sens_te)

    # measure the fairness of each model
    base_fairness = demographic_parity_difference(y_te, baseline_output, sensitive_features=sens_te)
    thresh_fairness = demographic_parity_difference(y_te, thresh_output, sensitive_features=sens_te)

    print('Base score: {}'.format(model.score(x_te, y_te)))
    print('Threshold score: {}'.format(1 - np.sum((np.abs(thresh_output - y_te))) / len(y_te)))

    print('Base fairness: {}'.format(base_fairness))
    print('Threshold fairness: {}'.format(thresh_fairness))

    print('stop')
