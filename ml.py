import time
import numpy as np
import scipy
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
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
    for line in open("2016HistoricalData.csv").read().split("\n"):
        line = line.split(",")
        if len(line) == 5:
            features.append(line[1:])
            classes.append(line[0])
    return np.array(features[1:], dtype=float), np.array(classes[1:], dtype=float)

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
    data = stratify(*getDataLabeled(), 0)
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
    for i, result in enumerate(results):
        if human: print(metrics[i].__name__ + ": " + str(result))
        else: print(result)
    return results

def train(name, model, param, X_train, X_test, y_train, y_test, human=True):
    print(name + ":")
    start_time = time.time()
    regr = RandomizedSearchCV(model, param, n_iter=100).fit(X_train, y_train)
    end_time = time.time()
    print("Time to train (seconds): " + str(end_time - start_time))
    return end_time - start_time, test(y_test, regr.predict(X_test), human)

data = getFinalData()
models = [svm.SVR(), MLPRegressor(max_iter=1000), DecisionTreeRegressor(), RandomForestRegressor()]
names = ["SVM", "NN", "DT", "RF"]
parameters = [{'C': scipy.stats.expon(scale=1),
               'gamma': scipy.stats.expon(scale=.1),
               'degree': scipy.stats.expon(scale=1),
               'coef0': scipy.stats.expon(scale=1),
               'tol': scipy.stats.expon(scale=.01),
               'kernel': ['rbf', 'linear'],
               'class_weight':['balanced', None],
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
              }]

t = []
y = [[] for i in range(4)]
avgs = [[0, 0, 0, 0] for i in range(4)]
n = 30
for k in range(n):
    for i in range(len(models)):
        out = train(names[i], models[i], parameters[i], *data)
        t.append(out[0])
        for j in range(4):
            y[j].append(out[1][j])
            avgs[j][i] += out[1][j]/n

for row in avgs: print(row)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.title('Performace of four models by four metrics compared to training time')
plt.xlabel('Time to train (seconds)')
plt.ylabel('Model performance', labelpad=40)

colors, markers = ['b', 'g', 'r', 'y'], ['.', 'o', '^', 's']
ylabels = ['Mean absolute error', 'Median absolute error', 'Explained variance score', r'$R^2$' + ' score']
for i, ax in enumerate(axes.flat):
    for j in range(len(y[i])):
        ax.scatter(t[j], y[i][j], c=colors[j % 4], marker=markers[j % 4], label=names[j % 4])
    ax.set_ylabel(ylabels[i])
    h, l = ax.get_legend_handles_labels()
    fig.legend(h[:4], l[:4], bbox_to_anchor=(0.95, 0.75), loc=1, borderaxespad=0.)

plt.tight_layout()
plt.savefig('graph.png', dpi=800, bbox_inches='tight')
plt.show()

#print(regr.predict(*getDataUnlabeled()))
#print(regr.cv_results_)
#print(regr.get_params().keys())
