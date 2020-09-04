import numpy as np
import os, sys, torch
sys.path.append('../')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from lifelines.utils import concordance_index   
sys.path.append('../')
sys.path.append('../../data/ml_mmrf')
sys.path.append('../../data/')
from sklearn.metrics import confusion_matrix
from ml_mmrf_v1.data import load_mmrf
from synthetic.synthetic_data import load_synthetic_data_trt, load_synthetic_data_noisy
from semi_synthetic.ss_data import *
from models.sfomm import SFOMM
from mord.threshold_based import LogisticAT 

def return_per_class_acc(y_true, y_pred): 
    accs = []
    for i in range(max(y_true)+1): 
        idxs = np.where(y_true == i)
        t = y_true[idxs]
        p = y_pred[idxs]
        acc = sum(t == p) / len(t)
        accs.append(acc)
    return accs

def setup_data(data, resample=True):
    X        = np.concatenate([data['train']['b'], data['train']['x'][:,0,:], data['train']['a'][:,0,:]], axis=-1)
    Y        = data['train']['ys_seq'][:,0]
    if resample: 
        ros = RandomOverSampler(random_state=0)
        smote = SMOTE(); ada = ADASYN()
        print('resampling...')
        X, Y = ros.fit_resample(X, Y)
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
    # auc    = roc_auc_score(Y_valid, y_pred, average='weighted', multi_class='ovr')
    auc = 0.
    accs = return_per_class_acc(Y_valid, y_pred)
    return f1, precision, recall, cf_matrix, auc, accs

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
    # auc    = roc_auc_score(Y_valid, y_pred, average='weighted', multi_class='ovr')
    auc = 0.    
    accs = return_per_class_acc(Y_valid, y_pred)
    return f1, precision, recall, cf_matrix, auc, accs

def get_mord_results(dataset, fold): 
    X, Y, S, S_oh, X_valid, Y_valid, CE_valid = setup_data(dataset[fold], resample=True)
    mo = LogisticAT(alpha=1.)
    mo.fit(X, Y)
    cutpoints = mo.theta_
    print(f'Cutpoints: {cutpoints}')
    y_pred = mo.predict(X_valid)
    print(f'predictions: {y_pred}')
    print(f'ground truth: {Y_valid}')
    cf_matrix = confusion_matrix(Y_valid, y_pred)
    f1     = f1_score(Y_valid, y_pred, average='weighted')
    precision = precision_score(Y_valid, y_pred, average='weighted')
    recall = recall_score(Y_valid, y_pred, average='weighted')
    # auc    = roc_auc_score(Y_valid, y_pred, average='weighted', multi_class='ovr')
    auc = 0.    
    accs = return_per_class_acc(Y_valid, y_pred)
    return f1, precision, recall, cf_matrix, auc, accs    

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
    f1_mo, p_mo, r_mo, mo_matrix, mo_auc, mo_accs = get_mord_results(ddata, fold=1)
    f1_lr, p_lr, r_lr, lr_matrix, lr_auc, lr_accs = get_lr_results(ddata, fold=1)
    rf_aucs = []; f1_rfs = []; p_rfs = []; r_rfs = []; rf_accs_list = []
    for i in range(5): 
        f1_rf, p_rf, r_rf, rf_matrix, rf_auc, rf_accs = get_rf_results(ddata, fold=1)
        rf_aucs.append(rf_auc)
        f1_rfs.append(f1_rf)
        p_rfs.append(p_rf)
        r_rfs.append(r_rf)
        rf_accs_list.append(rf_accs)
    acc0 = [x[0] for x in rf_accs_list]; acc1 = [x[1] for x in rf_accs_list]; acc2 = [x[2] for x in rf_accs_list]
    final_rf_accs = [np.mean(acc0), np.mean(acc1), np.mean(acc2)]
    print(f'Logistic Ordinal Regression: F1 - {f1_mo}, Precision - {p_mo}, Recall - {r_mo}, AUC - {mo_auc}, per_class_accs - {mo_accs}')
    print(f'Logistic Regression: F1 - {f1_lr}, Precision - {p_lr}, Recall - {r_lr}, AUC - {lr_auc}, per_class_accs - {lr_accs}')
    print(f'Random Forest: F1 - {np.mean(f1_rfs)}, Precision - {np.mean(p_rfs)}, Recall - {np.mean(r_rfs)}, AUC - {np.mean(rf_aucs)}, per_class_accs - {final_rf_accs}')
