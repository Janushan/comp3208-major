import pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pydot
from matplotlib import pyplot as plt
from heapq import nlargest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

prostate = pandas.read_csv("TCGA-PRAD.mirna.tsv.gz", compression='gzip', sep="\t", header=0, index_col=0)
stomach = pandas.read_csv("TCGA-STAD.mirna.tsv.gz", compression='gzip', sep="\t", header=0, index_col=0)

prostate = prostate.transpose()
prostate['type'] = 'prostate'

stomach = stomach.transpose()
stomach['type'] = 'stomach'

dataset = prostate.append(stomach)

y = dataset['type']
x = dataset.drop('type', axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=101)


tree = DecisionTreeClassifier(max_depth=5)

tree.fit(X_train, y_train)

prediction = tree.predict(X_test)

print(accuracy_score(y_test, prediction))



export_graphviz(tree,
                out_file="forest.dot",
                feature_names=x.columns.values,
                class_names=tree.classes_,
                rounded=True,
                filled=True
               )

(graph,) = pydot.graph_from_dot_file('forest.dot')
graph.write_png('somefile.png')

svm = SVC(kernel='linear', gamma=1, C=100)
svm.fit(X_train, y_train)

svm_prediction = svm.predict(X_test)

print(accuracy_score(y_test, svm_prediction))

coefs = np.ravel(svm.coef_)

k = 10
test = np.argpartition(coefs, len(coefs) - k)[-k:]

test = np.ravel(np.flip(test))

values = x.columns.values


forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

forest.fit(X_train, y_train)

forest_prediction = forest.predict(X_test)

print(accuracy_score(y_test, forest_prediction))

result = forest.feature_importances_

x = []
y = []

k = 10
test1 = np.argpartition(result, len(result) - k)[-k:]

test1 = np.ravel(np.flip(test))

for i in range(0, 5):

    x.append(coefs[test1[i]])
    y.append(values[test1[i]])

plt.barh(range(len(y)),x)
plt.yticks(range(len(y)),y)

plt.show()


