from aeon.file_formats.ts import load_from_tsfile_to_dataframep
from projection import random_projection
from classifier import one_nn_classifier

X_train, y_train = load_from_tsfile_to_dataframe("input/InsectSound_TRAIN.ts")
X_test, y_test = load_from_tsfile_to_dataframe("input/InsectSound_TEST.ts")

X_train = X_train.applymap(lambda x: x.tolist()).apply(lambda row: row[0], axis=1).tolist()
X_test = X_test.applymap(lambda x: x.tolist()).apply(lambda row: row[0], axis=1).tolist()

print("X_train (données) :")
print(X_train.head())

print("\ny_train (étiquettes) :")
print(y_train[:5])

print("X_test (données) :")
print(X_test.head())

print("\ny_test (étiquettes) :")
print(y_test[:5])

import numpy as np
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train_proj, X_test_proj = random_projection(X_train, X_test, output_dim=50, seed=42)
#output_dim est le k à choisir en se basant sur le lemme de Johnson-Lindenstrauss 

#acc = one_nn_classifier(X_train_proj, y_train, X_test_proj, y_test)

#print(f"Accuracy: {acc:.4f}")