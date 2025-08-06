import numpy as np
import matplotlib.pyplot as plt
from sktime.datasets import load_from_tsfile_to_dataframe
from pipeline import get_pipeline_from_data
from sklearn.model_selection import learning_curve

def process(path):
    X_df, y = load_from_tsfile_to_dataframe("data/train.ts")
    X_np = X_df.applymap(lambda x: x.tolist()).apply(lambda row: row[0], axis=1).tolist()#les séries doivent être univariées
    X_np = np.array(X_np)
    
    pipeline = get_pipeline_from_data(X_np, epsilon=0.25)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipeline,
        X=X_np,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        shuffle=True,
        random_state=42
    )

    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training accuracy")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation accuracy")
    plt.xlabel("Taille du jeu d'entraînement")
    plt.ylabel("Accuracy")
    plt.title("Courbe d'apprentissage : 1-NN + projection aléatoire")
    plt.legend()
    plt.grid()
    plt.show()