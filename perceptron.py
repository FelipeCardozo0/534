import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def read_file(filename):
    emails = list()
    label = list()
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split()
            label.append(int(row[0]))
            emails.append(row[1:])
    return emails, label

def build_vocab(train, test, minn):
    # 1. build vocabulary from train that appear in >= minn emails
    doc_freq = Counter()
    for email in train:
        doc_freq.update(set(email))
        
    vocab_dict = {}
    idx = 0
    for word, count in doc_freq.items():
        if count >= minn:
            vocab_dict[word] = idx
            idx += 1
            
    p = len(vocab_dict)
    n = len(train)
    m = len(test)
    
    # 2. transform train
    train_x = np.zeros((n, p))
    for i, email in enumerate(train):
        for word in email:
            if word in vocab_dict:
                train_x[i, vocab_dict[word]] = 1
                
    # transform test
    test_x = np.zeros((m, p))
    for i, email in enumerate(test):
        for word in email:
            if word in vocab_dict:
                test_x[i, vocab_dict[word]] = 1
                
    return train_x, test_x, vocab_dict

class Perceptron():
    def __init__(self, epoch):
        self.epoch = epoch
        self.w = None

    def get_weight(self):
        return self.w

    def sample_update(self, x, y):
        # dot product
        activation = np.dot(self.w, x)
        # prediction: if w.x >= 0, predict 1, else 0
        pred = 1 if activation >= 0 else 0
        mistake = 1 if pred != y else 0
        
        if mistake:
            # y=1, pred=0 -> add x
            # y=0, pred=1 -> sub x
            if y == 1:
                self.w = self.w + x
            else:
                self.w = self.w - x
                
        return self.w, mistake

    def train(self, trainx, trainy):
        n, p_plus_1 = trainx.shape
        if self.w is None:
            self.w = np.zeros(p_plus_1)
            
        mistakes_per_epoch = {}
        for e in range(1, self.epoch + 1):
            mistakes = 0
            for i in range(n):
                _, mistk = self.sample_update(trainx[i], trainy[i])
                mistakes += mistk
                
            mistakes_per_epoch[e] = mistakes
            if mistakes == 0:
                break
                
        return mistakes_per_epoch

    def predict(self, newx):
        # newx: m x p_plus_1
        activations = np.dot(newx, self.w)
        preds = np.where(activations >= 0, 1, 0)
        return preds

class AvgPerceptron(Perceptron):
    def __init__(self, epoch):
        super().__init__(epoch)
        self.avg_w = None
        
    def get_weight(self):
        return self.avg_w

    def train(self, trainx, trainy):
        n, p_plus_1 = trainx.shape
        if self.w is None:
            self.w = np.zeros(p_plus_1)
        if self.avg_w is None:
            self.avg_w = np.zeros(p_plus_1)
            
        mistakes_per_epoch = {}
        counter = 1
        for e in range(1, self.epoch + 1):
            mistakes = 0
            for i in range(n):
                _, mistk = self.sample_update(trainx[i], trainy[i])
                mistakes += mistk
                self.avg_w = self.avg_w + self.w
                counter += 1
                
            mistakes_per_epoch[e] = mistakes
            if mistakes == 0:
                # Need to update avg_w for the remaining epochs if we stop early?
                # Actually, averaged perceptron usually just returns the average up to the stopping point.
                break
                
        # Final average
        self.avg_w = self.avg_w / counter
        return mistakes_per_epoch
        
    def predict(self, newx):
        activations = np.dot(newx, self.avg_w)
        preds = np.where(activations >= 0, 1, 0)
        return preds

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    # Run Experiments for HW3
    emails, labels = read_file("spamAssassin.data")
    
    # 3a Validation Methodology
    # We split 80% Train, 20% Test. We further split Train into 75% SubTrain and 25% Validation
    # to plot generalization error (3g, 3k) and choose optimal params. Then test on Test set (3l).
    labels = np.array(labels)
    # create indices for splitting
    indices = np.arange(len(emails))
    X_tr_idx, X_test_idx, y_train_idx, y_test_idx = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)
    
    X_subtr_idx, X_val_idx, y_subtr, y_val = train_test_split(X_tr_idx, y_train_idx, test_size=0.25, random_state=42, stratify=y_train_idx)
    
    # List comprehension to slice
    train_emails = [emails[i] for i in X_tr_idx]
    test_emails = [emails[i] for i in X_test_idx]
    
    subtr_emails = [emails[i] for i in X_subtr_idx]
    val_emails = [emails[i] for i in X_val_idx]
    
    # 3b Build Vocab
    minn = 30
    # For training the final model:
    trainx, testx, vocab = build_vocab(train_emails, test_emails, minn)
    # Add bias column
    trainx = np.hstack([np.ones((trainx.shape[0], 1)), trainx])
    testx = np.hstack([np.ones((testx.shape[0], 1)), testx])
    
    # For hyperparameter tuning:
    subtr_x, val_x, vocab_sub = build_vocab(subtr_emails, val_emails, minn)
    subtr_x = np.hstack([np.ones((subtr_x.shape[0], 1)), subtr_x])
    val_x = np.hstack([np.ones((val_x.shape[0], 1)), val_x])
    
    # 3g Perceptron learning curve
    max_epochs = 30
    p = Perceptron(1) # We will train 1 epoch at a time to track validation error
    train_errors_p = []
    val_errors_p = []
    p.w = np.zeros(subtr_x.shape[1])
    mistakes_total = 0
    
    for e in range(1, max_epochs + 1):
        res = p.train(subtr_x, y_subtr)
        m = res[1]
        mistakes_total += m
        train_err = np.mean(p.predict(subtr_x) != y_subtr)
        val_err = np.mean(p.predict(val_x) != y_val)
        train_errors_p.append(train_err)
        val_errors_p.append(val_err)
        
    print(f"Perceptron Total mistakes over {max_epochs} epochs: {mistakes_total}")
    
    plt.plot(range(1, max_epochs + 1), train_errors_p, label='Train Error')
    plt.plot(range(1, max_epochs + 1), val_errors_p, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.title('Perceptron Error vs Epochs')
    plt.legend()
    plt.savefig('q3g_perceptron_curve.png')
    plt.close()
    
    # 3k Averaged Perceptron learning curve
    ap = AvgPerceptron(1)
    train_errors_ap = []
    val_errors_ap = []
    ap.w = np.zeros(subtr_x.shape[1])
    ap.avg_w = np.zeros(subtr_x.shape[1])
    
    counter = 1
    for e in range(1, max_epochs + 1):
        # manual epoch loop to collect stats at each epoch
        # we can't just call train(1) naively without handling counter, to be precise:
        mistakes = 0
        for i in range(len(subtr_x)):
            _, mistk = ap.sample_update(subtr_x[i], y_subtr[i])
            mistakes += mistk
            ap.avg_w = ap.avg_w + ap.w
            counter += 1
            
        # temporarily compute current average to test
        temp_avg_w = ap.avg_w / counter
        # prediction with current avg
        activations_tr = np.dot(subtr_x, temp_avg_w)
        preds_tr = np.where(activations_tr >= 0, 1, 0)
        activations_v = np.dot(val_x, temp_avg_w)
        preds_v = np.where(activations_v >= 0, 1, 0)
        
        train_errors_ap.append(np.mean(preds_tr != y_subtr))
        val_errors_ap.append(np.mean(preds_v != y_val))
        
    plt.plot(range(1, max_epochs + 1), train_errors_ap, label='Train Error')
    plt.plot(range(1, max_epochs + 1), val_errors_ap, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate')
    plt.title('Averaged Perceptron Error vs Epochs')
    plt.legend()
    plt.savefig('q3k_avg_perceptron_curve.png')
    plt.close()
    
    # 3l Optimal Model Training
    # Suppose Avg Perceptron after 30 epochs is best
    opt_clf = AvgPerceptron(max_epochs)
    opt_clf.train(trainx, labels[X_tr_idx])
    test_preds = opt_clf.predict(testx)
    test_err = np.mean(test_preds != labels[X_test_idx])
    print(f"3l Optimal AvgPerceptron Test Error: {test_err:.4f}")
    
    # 3m Top 15 positive and negative words
    w = opt_clf.get_weight()
    # first element is bias
    w_feat = w[1:]
    # invert vocabulary dict to map index to word
    idx_to_word = {idx: word for word, idx in vocab.items()}
    sorted_idx = np.argsort(w_feat)
    
    most_negative_idx = sorted_idx[:15]
    most_positive_idx = sorted_idx[-15:][::-1]
    
    print("Most Negative Words:")
    for i in most_negative_idx:
        print(f"{idx_to_word[i]}: {w_feat[i]:.4f}")
        
    print("\nMost Positive Words:")
    for i in most_positive_idx:
        print(f"{idx_to_word[i]}: {w_feat[i]:.4f}")
