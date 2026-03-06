import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def do_nothing(train, test):
    return train.copy(), test.copy()

def do_std(train, test):
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train)
    test_std = scaler.transform(test)
    return train_std, test_std

def do_log(train, test):
    train_log = np.log(train + 0.1)
    test_log = np.log(test + 0.1)
    return train_log, test_log

def do_bin(train, test):
    train_bin = (train > 0).astype(float)
    test_bin = (test > 0).astype(float)
    return train_bin, test_bin

def eval_nb(trainx, trainy, testx, testy):
    model = GaussianNB()
    model.fit(trainx, trainy)
    
    train_preds = model.predict(trainx)
    train_probs = model.predict_proba(trainx)[:, 1]
    
    test_preds = model.predict(testx)
    test_probs = model.predict_proba(testx)[:, 1]
    
    return {
        'train-acc': accuracy_score(trainy, train_preds),
        'train-auc': roc_auc_score(trainy, train_probs),
        'test-acc': accuracy_score(testy, test_preds),
        'test-auc': roc_auc_score(testy, test_probs),
        'test-prob': test_probs
    }

def eval_lr(trainx, trainy, testx, testy):
    # Depending on sklearn version, unregularized is penalty=None or penalty='none'
    try:
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
    except ValueError:
        model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000)
        
    model.fit(trainx, trainy)
    
    train_preds = model.predict(trainx)
    train_probs = model.predict_proba(trainx)[:, 1]
    
    test_preds = model.predict(testx)
    test_probs = model.predict_proba(testx)[:, 1]
    
    return {
        'train-acc': accuracy_score(trainy, train_preds),
        'train-auc': roc_auc_score(trainy, train_probs),
        'test-acc': accuracy_score(testy, test_preds),
        'test-auc': roc_auc_score(testy, test_probs),
        'test-prob': test_probs
    }

if __name__ == "__main__":
    # Load data
    train_data = np.loadtxt('spam/spam.train.dat')
    test_data = np.loadtxt('spam/spam.test.dat')
    
    trainy = train_data[:, -1]
    trainx = train_data[:, :-1]
    
    testy = test_data[:, -1]
    testx = test_data[:, :-1]
    
    preprocessors = {
        'None': do_nothing,
        'Standardized': do_std,
        'Log Transformed': do_log,
        'Binarized': do_bin
    }
    
    nb_results = {}
    lr_results = {}
    
    print("-" * 50)
    print("Naive Bayes Results")
    print(f"{'Preprocessing':<20} | {'Train Acc':<10} | {'Train AUC':<10} | {'Test Acc':<10} | {'Test AUC':<10}")
    print("-" * 50)
    for name, func in preprocessors.items():
        tx, tex = func(trainx, testx)
        res = eval_nb(tx, trainy, tex, testy)
        nb_results[name] = res
        print(f"{name:<20} | {res['train-acc']:.4f}     | {res['train-auc']:.4f}     | {res['test-acc']:.4f}    | {res['test-auc']:.4f}")
        
    print("\n" + "-" * 50)
    print("Logistic Regression Results")
    print(f"{'Preprocessing':<20} | {'Train Acc':<10} | {'Train AUC':<10} | {'Test Acc':<10} | {'Test AUC':<10}")
    print("-" * 50)
    for name, func in preprocessors.items():
        tx, tex = func(trainx, testx)
        res = eval_lr(tx, trainy, tex, testy)
        lr_results[name] = res
        print(f"{name:<20} | {res['train-acc']:.4f}     | {res['train-auc']:.4f}     | {res['test-acc']:.4f}    | {res['test-auc']:.4f}")

    # Plot 1: 4 NB curves
    plt.figure(figsize=(8, 6))
    for name, res in nb_results.items():
        fpr, tpr, _ = roc_curve(testy, res['test-prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['test-auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Naive Bayes')
    plt.legend()
    plt.savefig('roc_nb.png')
    plt.close()
    
    # Plot 2: 4 LR curves
    plt.figure(figsize=(8, 6))
    for name, res in lr_results.items():
        fpr, tpr, _ = roc_curve(testy, res['test-prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['test-auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Logistic Regression')
    plt.legend()
    plt.savefig('roc_lr.png')
    plt.close()
    
    # Find best NB and best LR by test AUC
    best_nb_name = max(nb_results, key=lambda k: nb_results[k]['test-auc'])
    best_lr_name = max(lr_results, key=lambda k: lr_results[k]['test-auc'])
    
    # Plot 3: Best NB and Best LR
    plt.figure(figsize=(8, 6))
    
    fpr_nb, tpr_nb, _ = roc_curve(testy, nb_results[best_nb_name]['test-prob'])
    plt.plot(fpr_nb, tpr_nb, label=f"NB - {best_nb_name} (AUC = {nb_results[best_nb_name]['test-auc']:.3f})")
    
    fpr_lr, tpr_lr, _ = roc_curve(testy, lr_results[best_lr_name]['test-prob'])
    plt.plot(fpr_lr, tpr_lr, label=f"LR - {best_lr_name} (AUC = {lr_results[best_lr_name]['test-auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Best Naive Bayes vs Best Logistic Regression')
    plt.legend()
    plt.savefig('roc_best.png')
    plt.close()
    
    print("\nBest NB Model:", best_nb_name)
    print("Best LR Model:", best_lr_name)
