from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
import matplotlib.pyplot as plt
import numpy as np

def tune_dt(x, y, dparams, lsparams):
    """
    Determines optimal max_depth and min_samples_leaf based on Grid Search and AUC.
    Returns: {"best-depth": ..., "best-leaf-samples": ..., "grid_results": ...}
    x and y should be the training + validation set. We'll use GridSearchCV with 5-fold CV.
    """
    param_grid = {'max_depth': dparams, 'min_samples_leaf': lsparams}
    dt = DecisionTreeClassifier(random_state=42)
    clf = GridSearchCV(dt, param_grid, scoring='roc_auc', cv=5, return_train_score=False)
    clf.fit(x, y)
    
    return {
        "best-depth": clf.best_params_['max_depth'],
        "best-leaf-samples": clf.best_params_['min_samples_leaf'],
        "best-auc": clf.best_score_,
        "grid_results": clf.cv_results_
    }

if __name__ == "__main__":
    from preprocess import preprocess_and_partition, feature_selection_pearson, feature_selection_spearman, feature_selection_mi
    
    # Reloading with partitioning
    X_train, y_train, X_val, y_val, X_test, y_test, feats = preprocess_and_partition('loan_default.csv')
    
    # For tune_dt, we will combine Train and Val and use CV, OR we can just use train.
    # The instructions say "assume test has already been set aside".
    # Since we did 70/15/15, let's just concatenate Train+Val for a 85% train set for CV
    X_tv = np.vstack((X_train, X_val))
    y_tv = np.concatenate((y_train, y_val))
    
    dparams = [2, 3, 4, 5, 7, 10, 15]
    lsparams = [1, 2, 5, 10, 20, 50, 100]
    
    res = tune_dt(X_tv, y_tv, dparams, lsparams)
    print("Optimal Params:", res["best-depth"], res["best-leaf-samples"], res["best-auc"])
    
    # Plot validation AUC
    results = res["grid_results"]
    depths = results['param_max_depth'].data.astype(int)
    leaves = results['param_min_samples_leaf'].data.astype(int)
    aucs = results['mean_test_score']
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(depths, leaves, s=(aucs-0.5)*1000, c=aucs, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Validation AUC')
    plt.xlabel('Max Depth')
    plt.ylabel('Min Samples Leaf')
    plt.title('Validation AUC by Max Depth and Min Samples Leaf')
    plt.grid(True)
    plt.savefig('q2b_plot.png')
    plt.close()
    
    # 2c Re-train on entire training data (X_tv)
    dt_opt = DecisionTreeClassifier(max_depth=res["best-depth"], min_samples_leaf=res["best-leaf-samples"], random_state=42)
    dt_opt.fit(X_tv, y_tv)
    
    y_pred_probs = dt_opt.predict_proba(X_test)[:, 1]
    y_pred = dt_opt.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    print(f"2c: Test AUC: {auc:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
    
    # 2d Visualization of top 3 levels
    plt.figure(figsize=(20, 10))
    plot_tree(dt_opt, max_depth=3, feature_names=feats, class_names=['Paid', 'Default'], filled=True, rounded=True)
    plt.savefig('q2d_tree.png')
    plt.close()
    
    # 2e Analyze effects of filtering methods
    print("--- 2e Feature Selection Analysis ---")
    k = 10 # choose top 10 features, for example, or top 5
    # Let's use top 5 Features for all
    k = 5
    
    # Pearson
    p_idx = feature_selection_pearson(X_tv, y_tv)
    X_tv_p = X_tv[:, p_idx[:k]]
    X_test_p = X_test[:, p_idx[:k]]
    res_p = tune_dt(X_tv_p, y_tv, dparams, lsparams)
    dt_p = DecisionTreeClassifier(max_depth=res_p["best-depth"], min_samples_leaf=res_p["best-leaf-samples"], random_state=42)
    dt_p.fit(X_tv_p, y_tv)
    pred_p = dt_p.predict(X_test_p)
    prob_p = dt_p.predict_proba(X_test_p)[:, 1]
    print(f"Pearson (Top {k}): AUC={roc_auc_score(y_test, prob_p):.4f}, F1={f1_score(y_test, pred_p):.4f}, F2={fbeta_score(y_test, pred_p, beta=2):.4f}")
    
    # Spearman
    s_idx = feature_selection_spearman(X_tv, y_tv)
    X_tv_s = X_tv[:, s_idx[:k]]
    X_test_s = X_test[:, s_idx[:k]]
    res_s = tune_dt(X_tv_s, y_tv, dparams, lsparams)
    dt_s = DecisionTreeClassifier(max_depth=res_s["best-depth"], min_samples_leaf=res_s["best-leaf-samples"], random_state=42)
    dt_s.fit(X_tv_s, y_tv)
    pred_s = dt_s.predict(X_test_s)
    prob_s = dt_s.predict_proba(X_test_s)[:, 1]
    print(f"Spearman (Top {k}): AUC={roc_auc_score(y_test, prob_s):.4f}, F1={f1_score(y_test, pred_s):.4f}, F2={fbeta_score(y_test, pred_s, beta=2):.4f}")
    
    # MI
    m_idx = feature_selection_mi(X_tv, y_tv)
    X_tv_m = X_tv[:, m_idx[:k]]
    X_test_m = X_test[:, m_idx[:k]]
    res_m = tune_dt(X_tv_m, y_tv, dparams, lsparams)
    dt_m = DecisionTreeClassifier(max_depth=res_m["best-depth"], min_samples_leaf=res_m["best-leaf-samples"], random_state=42)
    dt_m.fit(X_tv_m, y_tv)
    pred_m = dt_m.predict(X_test_m)
    prob_m = dt_m.predict_proba(X_test_m)[:, 1]
    print(f"MI (Top {k}): AUC={roc_auc_score(y_test, prob_m):.4f}, F1={f1_score(y_test, pred_m):.4f}, F2={fbeta_score(y_test, pred_m, beta=2):.4f}")
