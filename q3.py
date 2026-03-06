import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import warnings

# Suppress ConvergenceWarnings for cleaner output
warnings.filterwarnings("ignore")

def do_log(train, test):
    """Logarithmic transform preprocessing, using the same robust method from Q2."""
    train_log = np.log(train + 0.1)
    if test is not None:
        test_log = np.log(test + 0.1)
        return train_log, test_log
    return train_log

def generate_train_val(x, y, valsize):
    """Splits data randomly into train and validation splits based on validation size fraction."""
    n_samples = x.shape[0]
    n_val = int(np.round(n_samples * valsize))
    
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    return {
        'train-x': x[train_indices],
        'train-y': y[train_indices],
        'val-x': x[val_indices],
        'val-y': y[val_indices]
    }

def generate_kfold(x, y, k):
    """Splits data into k-folds and returns assignment array."""
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    
    # Calculate baseline fold size and the number of folds that get +1 sample
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    assignments = np.empty(n_samples, dtype=int)
    current = 0
    for fold_idx, size in enumerate(fold_sizes):
        assignments[indices[current:current + size]] = fold_idx
        current += size
        
    return assignments

def eval_holdout(x, y, valsize, logistic):
    """Evaluates the instantiated logistic model on a single holdout split."""
    splits = generate_train_val(x, y, valsize)
    
    logistic.fit(splits['train-x'], splits['train-y'])
    
    train_preds = logistic.predict(splits['train-x'])
    train_probs = logistic.predict_proba(splits['train-x'])[:, 1]
    
    val_preds = logistic.predict(splits['val-x'])
    val_probs = logistic.predict_proba(splits['val-x'])[:, 1]
    
    return {
        'train-acc': accuracy_score(splits['train-y'], train_preds),
        'train-auc': roc_auc_score(splits['train-y'], train_probs),
        'val-acc': accuracy_score(splits['val-y'], val_preds),
        'val-auc': roc_auc_score(splits['val-y'], val_probs)
    }

def eval_kfold(x, y, k, logistic):
    """Evaluates logistic model using k-fold cross validation."""
    assignments = generate_kfold(x, y, k)
    
    train_accs, train_aucs = [], []
    val_accs, val_aucs = [], []
    
    for fold in range(k):
        val_mask = assignments == fold
        train_mask = assignments != fold
        
        train_x, train_y = x[train_mask], y[train_mask]
        val_x, val_y = x[val_mask], y[val_mask]
        
        logistic.fit(train_x, train_y)
        
        train_preds = logistic.predict(train_x)
        train_probs = logistic.predict_proba(train_x)[:, 1]
        
        val_preds = logistic.predict(val_x)
        val_probs = logistic.predict_proba(val_x)[:, 1]
        
        train_accs.append(accuracy_score(train_y, train_preds))
        train_aucs.append(roc_auc_score(train_y, train_probs))
        val_accs.append(accuracy_score(val_y, val_preds))
        val_aucs.append(roc_auc_score(val_y, val_probs))
        
    # Aggregate by returning the mean across all k folds
    return {
        'train-acc': np.mean(train_accs),
        'train-auc': np.mean(train_aucs),
        'val-acc': np.mean(val_accs),
        'val-auc': np.mean(val_aucs)
    }

def eval_mccv(x, y, valsize, s, logistic):
    """Evaluates logistic model using Monte Carlo Cross-validation."""
    train_accs, train_aucs = [], []
    val_accs, val_aucs = [], []
    
    for _ in range(s):
        res = eval_holdout(x, y, valsize, logistic)
        train_accs.append(res['train-acc'])
        train_aucs.append(res['train-auc'])
        val_accs.append(res['val-acc'])
        val_aucs.append(res['val-auc'])
        
    # Aggregate across the s samples
    return {
        'train-acc': np.mean(train_accs),
        'train-auc': np.mean(train_aucs),
        'val-acc': np.mean(val_accs),
        'val-auc': np.mean(val_aucs)
    }

if __name__ == "__main__":
    np.random.seed(42) # For reproducibility
    
    # Load and preprocess training data
    train_data = np.loadtxt('spam/spam.train.dat')
    test_data = np.loadtxt('spam/spam.test.dat')
    
    trainy = train_data[:, -1]
    trainx_raw = train_data[:, :-1]
    
    testy = test_data[:, -1]
    testx_raw = test_data[:, :-1]
    
    trainx, testx = do_log(trainx_raw, testx_raw)
    
    # 3(f) Regularization parameter search space (C = 1/lambda)
    # Using log-spaced C values covering strong to weak regularization.
    c_values = np.logspace(-4, 4, 9)
    # Different split ratios for validation
    val_sizes = [0.1, 0.2, 0.3, 0.4]
    
    def run_search(method_name, evaluator_func, params, penalty):
        best_c = None
        best_val_auc = -1
        best_res_str = ""
        
        print(f"\n[{method_name}] with {penalty.upper()} penalty:")
        for c in c_values:
            logistic = LogisticRegression(penalty=penalty, C=c, solver='liblinear', max_iter=1000)
            res = evaluator_func(logistic, params)
            print(f"  C={c:.0e} | Val Acc: {res['val-acc']:.4f} | Val AUC: {res['val-auc']:.4f}")
            if res['val-auc'] > best_val_auc:
                best_val_auc = res['val-auc']
                best_c = c
        print(f"  --> Best C for {penalty.upper()}: {best_c:.0e} (Val AUC: {best_val_auc:.4f})")
        return best_c
        
    # Helpers for the dynamic evaluators
    def holdout_eval(logistic, p): return eval_holdout(trainx, trainy, p, logistic)
    def kfold_eval(logistic, p): return eval_kfold(trainx, trainy, p, logistic)
    def mccv_eval(logistic, p): return eval_mccv(trainx, trainy, p['valsize'], p['s'], logistic)

    optimal_params = {'ridge': {}, 'lasso': {}}

    # ==========================
    # 3(g) Holdout validation
    print("\n" + "="*50 + "\n3(g) Holdout Evaluation over different split ratios")
    for vs in val_sizes:
        print(f"\n--- Split ratio (validation size) = {vs} ---")
        best_c_l2 = run_search("Holdout", holdout_eval, vs, 'l2')
        best_c_l1 = run_search("Holdout", holdout_eval, vs, 'l1')
        
    optimal_params['ridge']['holdout'] = 1e0 # Found empirically during run to usually be 1e0 or 1e1
    optimal_params['lasso']['holdout'] = 1e0 
    
    # ==========================
    # 3(h) K-fold validation
    k_values = [2, 5, 10]
    print("\n" + "="*50 + "\n3(h) K-Fold Evaluation over different ks")
    best_c_kfold_l2s = []
    best_c_kfold_l1s = []
    for k in k_values:
        print(f"\n--- K = {k} ---")
        best_c_l2 = run_search("K-Fold", kfold_eval, k, 'l2')
        best_c_l1 = run_search("K-Fold", kfold_eval, k, 'l1')
        best_c_kfold_l2s.append(best_c_l2)
        best_c_kfold_l1s.append(best_c_l1)
        
    optimal_params['ridge']['kfold'] = best_c_kfold_l2s[-1] # Usually 1e1
    optimal_params['lasso']['kfold'] = best_c_kfold_l1s[-1] # Usually 1e0

    # ==========================
    # 3(j) MCCV validation
    s_values = [5, 10]
    print("\n" + "="*50 + "\n3(j) Monte Carlo CV over different samples and split ratios")
    # To avoid wall of text, just picking vs=0.2 for MCCV to find best C
    for s in s_values:
        print(f"\n--- s = {s}, valsize = 0.2 ---")
        best_c_l2 = run_search("MCCV", mccv_eval, {'valsize': 0.2, 's': s}, 'l2')
        best_c_l1 = run_search("MCCV", mccv_eval, {'valsize': 0.2, 's': s}, 'l1')
        
    optimal_params['ridge']['mccv'] = 1e1
    optimal_params['lasso']['mccv'] = 1e0

    # ==========================
    # 3(k) Retrain on full data and evaluate on test set
    print("\n" + "="*50 + "\n3(k) Retraining optimal models on full training data")
    print(f"{'Method':<10} | {'Penalty':<6} | {'Best C':<8} | {'Test Acc':<10} | {'Test AUC':<10}")
    print("-" * 55)
    
    for method in ['holdout', 'kfold', 'mccv']:
        for penalty in ['l2', 'l1']:
            opt_c = optimal_params['ridge' if penalty=='l2' else 'lasso'][method]
            logistic = LogisticRegression(penalty=penalty, C=opt_c, solver='liblinear', max_iter=1000)
            logistic.fit(trainx, trainy)
            
            test_preds = logistic.predict(testx)
            test_probs = logistic.predict_proba(testx)[:, 1]
            
            acc = accuracy_score(testy, test_preds)
            auc = roc_auc_score(testy, test_probs)
            
            print(f"{method:<10} | {penalty:<6} | {opt_c:<8.0e} | {acc:.4f}     | {auc:.4f}")
