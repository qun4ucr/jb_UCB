import matplotlib.pyplot as plt
import seaborn as sns, numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, recall_score, confusion_matrix, \
    classification_report

creditcardDF = pd.read_csv('./CC.csv')

# creditcardDF = pd.read_csv('./CC.csv')
creditcardDF = creditcardDF.loc[:, ~creditcardDF.columns.str.contains('^Unnamed')]
# Log transformation for skewed dataset
creditcardDF['Amount'] = np.log(creditcardDF['Amount'] + 1)
creditcardDF['Time'] = np.log(creditcardDF['Time'] + 1)
# Get label
y = creditcardDF['Class']
# Get features
X = creditcardDF.drop(['Class'], axis=1)
# Stratified sampling
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=6)
print('training data has ' + str(X_train.shape[0]) + ' observations with ' + str(X_train.shape[1]) + ' features')
print('test data has ' + str(X_test.shape[0]) + ' observations with ' + str(X_test.shape[1]) + ' features')


class SVM:
    def __init__(self, kernel='linear', C=10000.0, max_iter=5000, degree=3, gamma=1):
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.C = C
        self.max_iter = max_iter

    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def pre_process(self,tx, ty):
        dropout_tx, dropout_ty = [], []
        bal = int(len(ty) / sum(ty))
        for i in range(len(tx)):
            if ty[i] == 1:
                dropout_tx.append(tx[i])
                dropout_ty.append(ty[i])
            else:
                if i % int(20) == 0:  # much more balanced
                    dropout_tx.append(tx[i])
                    dropout_ty.append(ty[i])

        dropout_tx, dropout_ty = np.array(dropout_tx), np.array(dropout_ty)
        return (dropout_ty, dropout_ty)

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y * 2 - 1
        self.lambdas = np.zeros_like(self.y, dtype=float)
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        for _ in range(self.max_iter):
            if _ % 300 == 0:
                print("now iteration", _)
            for idxM in range(len(self.lambdas)):
                idxL = np.random.randint(0, len(self.lambdas))
                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                v0 = self.lambdas[[idxM, idxL]]
                k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)
                u = np.array([-self.y[idxL], self.y[idxM]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                self.lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)

        idx, = np.nonzero(self.lambdas > 1E-15)

        self.b = np.sum((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx]) / len(idx)

    def decision_function(self, X):
        print(self.X.shape,X.shape)
        res = np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b
        return res




ty = np.array(y.values)
tx = np.array(pd.DataFrame(X, columns=["V" + str(i) for i in range(1, 29)] + ["Amount", "Time"]))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, stratify = y, random_state=6)
train_ty = np.array(y_train.values)
train_tx = np.array(pd.DataFrame(X_train, columns=["V" + str(i) for i in range(1, 29)] + ["Amount", "Time"]))
test_ty = np.array(y_test.values)
test_tx = np.array(X_test.values)



print(sum(train_ty), len(train_ty),int(len(train_ty) / sum(train_ty)))
dropout_tx, dropout_ty = [], []
bal = int(len(train_ty) / sum(train_ty))


for i in range(len(train_tx)):
    if ty[i] == 1:
        dropout_tx.append(tx[i])
        dropout_ty.append(ty[i])
    else:
        if i % 10 == 0:  # much more balanced
            dropout_tx.append(tx[i])
            dropout_ty.append(ty[i])

dropout_tx, dropout_ty = np.array(dropout_tx), np.array(dropout_ty)

clf = SVM(max_iter=900,kernel="linear")
clf.fit(dropout_tx, dropout_ty)
result = clf.decision_function(test_tx)

def eval(norm_result,test_ty):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(norm_result)):
        if norm_result[i] == test_ty[i]:
            if norm_result[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if norm_result[i] == 1:
                fp += 1
            else:
                fn += 1

    auc = roc_auc_score(test_ty,norm_result)
    f1 = tp / (tp + 0.5 * (fp + fn))
    print("tp:", tp, " fp:", fp, "  tn:", tn, "fn:", fn," auc:",auc," f1:",f1)
    print(sum(test_ty))


norm_result = [ 1 if c>-0.001 else 0 for c in result  ]
eval(norm_result,test_ty)
norm_result = [ 1 if c>0 else 0 for c in result  ]
eval(norm_result,test_ty)
norm_result = [ 1 if c>-1 else 0 for c in result  ]
eval(norm_result,test_ty)
norm_result = [ 1 if c>1 else 0 for c in result  ]
eval(norm_result,test_ty)

