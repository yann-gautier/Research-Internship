from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def one_nn_classifier(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)
