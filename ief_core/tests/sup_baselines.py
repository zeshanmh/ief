import numpy as np
import os, sys, torch
sys.path.append('../')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from lifelines.utils import concordance_index   
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from sklearn.metrics import confusion_matrix
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from semi_synthetic.ss_data import *
from models.sfomm import SFOMM

def setup_data(data):
    X        = np.concatenate([data['train']['b'], data['train']['x'][:,0,:], data['train']['a'][:,0,:]], axis=-1)
    Y        = data['train']['ys_seq'][:,0]
    X_valid  = np.concatenate([data['valid']['b'], data['valid']['x'][:,0,:], data['valid']['a'][:,0,:]], axis=-1)
    Y_valid  = data['valid']['ys_seq'][:,0]
    CE_valid = data['valid']['ce']
    S, S_oh  = None, None
    if 'subtype' in data['train']:
        S       = data['train']['subtype']
        S_oh    = data['train']['subtype_oh']
    return X, Y, S, S_oh, X_valid, Y_valid, CE_valid

def get_lr_results(dataset, fold):
    X, Y, S, S_oh, X_valid, Y_valid, CE_valid = setup_data(dataset[fold])
    reg    = LogisticRegression(fit_intercept=False, max_iter=1000, class_weight='balanced').fit(X, Y)
    y_pred = reg.predict(X_valid)
    print(f'predictions: {y_pred}')
    print(f'ground truth: {Y_valid}')
    cf_matrix = confusion_matrix(Y_valid, y_pred)
    f1     = f1_score(Y_valid, y_pred, average='weighted')
    precision = precision_score(Y_valid, y_pred, average='weighted')
    recall = recall_score(Y_valid, y_pred, average='weighted')
    auc    = roc_auc_score(Y_valid, y_pred, average='weighted')
    return f1, precision, recall, cf_matrix, auc

def get_rf_results(dataset, fold):
    X, Y, S, S_oh, X_valid, Y_valid, CE_valid = setup_data(dataset[fold])
    nlreg  = RandomForestClassifier(n_estimators=1000, max_features=X.shape[1], class_weight='balanced')
    nlreg.fit(X, Y)
    y_pred = nlreg.predict(X_valid)
    print(f'predictions: {y_pred}')
    print(f'ground truth: {Y_valid}')
    cf_matrix = confusion_matrix(Y_valid, y_pred)
    f1     = f1_score(Y_valid, y_pred, average='weighted')
    precision = precision_score(Y_valid, y_pred, average='weighted')
    recall = recall_score(Y_valid, y_pred, average='weighted')
    auc    = roc_auc_score(Y_valid, y_pred, average='weighted')
    return f1, precision, recall, cf_matrix, auc

def get_oracle_results(dataset, fold):
    X, Y, S, S_oh, X_valid, Y_valid, CE_valid = setup_data(dataset[fold])
    assert S is not None and S_oh is not None,'expecting S/S_oh'
    base2sub  = RandomForestRegressor(n_estimators=1000, max_features=X.shape[1])
    base2sub.fit(X, S)
    sub2ys    = LinearRegression(fit_intercept=False).fit(S_oh, Y)
    sub_preds = base2sub.predict(X_valid)
    s_pred    = preds_to_category(sub_preds)
    s_pred_oh = make_one_hot(s_pred[:,None], n_values = 4)
    y_pred    = sub2ys.predict(s_pred_oh)
    mse = mean_squared_error(Y_valid, y_pred)
    r2  = r2_score(Y_valid, y_pred)
    ci  = concordance_index(Y_valid.ravel(), y_pred.ravel(), (1.-CE_valid).ravel())
    return mse, ci, r2


if __name__=='__main__':
    ddata = load_mmrf(fold_span = [1], digitize_K = 0, digitize_method = 'uniform', suffix='_2mos_tr')
    import pdb; pdb.set_trace()
    f1_lr, p_lr, r_lr, lr_matrix, lr_auc = get_lr_results(ddata, fold=1)
    f1_rf, p_rf, r_rf, rf_matrix, rf_auc = get_rf_results(ddata, fold=1)
    print(f'Logistic Regression: F1 - {f1_lr}, Precision - {p_lr}, Recall - {r_lr}, AUC - {lr_auc}')
    print(f'Random Forest: F1 - {f1_rf}, Precision - {p_rf}, Recall - {r_rf}, AUC - {rf_auc}')
