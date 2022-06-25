import data_util.data_util as data_util
import numpy as np
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer

if __name__ == '__main__':
    # get the causality model data
    data, target, sens = data_util.gen_data(t=1)

    # get column labels
    dis_labels, con_labels, cat = data_util.get_data_type(data)

    # normalise the data
    data = data_util.normalise_data(data, dis_labels, con_labels)

    # normalise the target
    target = target.astype('category').cat.codes

    # save sensitive column, but remove it from the data
    sensitive = data['D']
    data = data.drop('D', axis=1)

    # fit model to training data
    model = LogisticRegression(max_iter=10000)
    model.fit(data, target)

    # fit threshold optimiser to model
    thresh_model = ThresholdOptimizer(
            estimator=model,
            constraints='demographic_parity',
            objective='accuracy_score',
            prefit=True,
            grid_size=100000)
    thresh_model.fit(data, target, sensitive_features=sensitive)

    # create testing sample
    data_te, target_te, sens_te = data_util.gen_data(t=1)
    # get column labels
    dis_labels_te, con_labels_te, cat_te = data_util.get_data_type(data_te)
    # normalise the data
    data_te = data_util.normalise_data(data_te, dis_labels_te, con_labels_te)
    # normalise the target
    target_te = target_te.astype('category').cat.codes
    # save sensitive column, but remove it from the data
    sensitive_te = data_te['D']
    data_te = data_te.drop('D', axis=1)

    # make predictions on testing data set
    baseline_output = model.predict(data_te)
    thresh_output = thresh_model.predict(data_te, sensitive_features=sensitive_te)

    # measure the fairness of each model
    base_fairness = demographic_parity_difference(target_te, baseline_output, sensitive_features=sensitive_te)
    thresh_fairness = demographic_parity_difference(target_te, thresh_output, sensitive_features=sensitive_te)

    print('Base score: {}'.format(model.score(data_te, target_te)))
    print('Threshold score: {}'.format(1 - np.sum((np.abs(thresh_output - target_te))) / len(target_te)))

    print('Base fairness: {}'.format(base_fairness))
    print('Threshold fairness: {}'.format(thresh_fairness))
