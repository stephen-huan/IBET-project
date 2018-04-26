import time
import numpy as np
import scipy
from scipy import stats
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt

def getDataUnlabeled():
    features = []
    for line in open("test.csv").read().split("\n"):
        line = line.split(",")
        if len(line) == 4: features.append(line)
    return np.array(features, dtype=float)

def getDataLabeled():
    features, classes = [], []
    for line in open("Final.csv").read().split("\n")[1:]:
        line = line.split(",")
        if len(line) > 1:
            cell = [x if x != "" else np.nan for x in line[:-1]]
            cell.insert(0, ord(cell[0][0]))
            cell[1] = cell[1][1:]
            features.append(cell)
            eggs = line[-1] if line[-1] != "" else np.nan
            classes.append(eggs)
    mean = sum([int(x) if x == x else 0 for x in classes])/len(classes)
    return Imputer(missing_values='NaN', strategy='mean', axis=0).fit(features).transform(features), np.array([x if x == x else mean for x in classes], dtype=float)

def stratify(data_X, data_y, feature):
    out_X = {}
    out_y = {}
    for i, row in enumerate(data_X):
        index = row[feature]
        if index in out_X:
            out_X[index] = np.concatenate((out_X[index], [row]))
            out_y[index] = np.append(out_y[index], data_y[i])
        else:
            out_X[index] = np.array([row])
            out_y[index] = np.array(data_y[i])
    return out_X, out_y

def getFinalData():
    data = stratify(*getDataLabeled(), 2)
    i = iter(data[0])
    temp = next(i)
    final = train_test_split(data[0][temp], data[1][temp], test_size=0.2)
    for key in i:
        temp = train_test_split(data[0][key], data[1][key], test_size=0.2)
        for j in range(4):
            final[j] = np.concatenate((final[j], temp[j]))
    scaler = StandardScaler()
    scaler.fit(final[0])
    final[0] = scaler.transform(final[0])
    final[1] = scaler.transform(final[1])
    return final

def test(y_test, y_pred, human):
    metrics = [mean_absolute_error, median_absolute_error, explained_variance_score, r2_score]
    results = [metric(y_test, y_pred) for metric in metrics]
    if human:
        for i, result in enumerate(results):
            print(metrics[i].__name__ + ": " + str(result))
    return results

def train(name, model, param, X_train, X_test, y_train, y_test, human=True):
    start_time = time.time()
    regr = RandomizedSearchCV(model, param, n_iter=100) if len(param) > 0 else model
    regr.fit(X_train, y_train)
    end_time = time.time()
    if human: print(name + ":\n" + "Time to train (seconds): " + str(end_time - start_time))
    return end_time - start_time, test(y_test, regr.predict(X_test), human)

models = [svm.SVR(), MLPRegressor(max_iter=1000), DecisionTreeRegressor(), RandomForestRegressor(), LinearRegression()]
names = ["SVM", "NN", "DT", "RF", "LR"]
parameters = [{'C': scipy.stats.expon(scale=1),
               'gamma': scipy.stats.expon(scale=.1),
               'degree': scipy.stats.expon(scale=1),
               'coef0': scipy.stats.expon(scale=1),
               'tol': scipy.stats.expon(scale=.01),
               'kernel': ['rbf', 'linear'],
              },
              {'activation': ['logistic', 'relu'],
               'solver': ['sgd', 'lbfgs'],
               'alpha': scipy.stats.expon(scale=.001),
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'tol': scipy.stats.expon(scale=.01),
              },
              {'max_depth': scipy.stats.expon(scale=100),
              },
              {'max_depth': scipy.stats.expon(scale=100),
              },
              {}]

ln = len(models)
t = []
y = [[] for i in range(ln)]
nums = [[[] for j in range(ln)] for i in range(4)]
n = 30
for k in range(n):
    for i in range(ln):
        out = train(names[i], models[i], parameters[i], *getFinalData(), False)
        t.append(out[0])
        for j in range(4):
            y[j].append(out[1][j])
            nums[j][i].append(out[1][j])

for row in nums:
    mean = std = ""
    for arr in row:
        mean += str(stats.tmean(arr)) + " "
        std += str(stats.tstd(arr)) + " "
    print("Mean " + mean)
    print("STD " + std)
    print(stats.f_oneway(*row))
    print()

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Time to train (seconds)')
plt.ylabel('Model performance', labelpad=40)

colors, markers = ['b', 'g', 'r', 'y', 'm'], ['.', 'o', '^', 's', 'p']
ylabels = ['Mean absolute error', 'Median absolute error', 'Explained variance score', r'$R^2$' + ' score']
for i, ax in enumerate(axes.flat):
    for j in range(len(y[i])):
        ax.scatter(t[j], y[i][j], c=colors[j % ln], marker=markers[j % ln], label=names[j % ln])
    ax.set_ylabel(ylabels[i])
    h, l = ax.get_legend_handles_labels()
    fig.legend(h[:ln], l[:ln], bbox_to_anchor=(0.95, 0.75), loc=1, borderaxespad=0.)

plt.tight_layout()
plt.savefig('graph.png', dpi=800)
plt.show()

#print(regr.predict(*getDataUnlabeled()))
#print(regr.cv_results_)
#print(regr.get_params().keys())
