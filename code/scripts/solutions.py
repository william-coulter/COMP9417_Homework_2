import argparse
from statistics import mean

from numpy import linspace
from pandas import read_csv, concat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, make_scorer

import matplotlib.pyplot as plt

Q1_DATA_DIR = "./data/Q1.csv"

### UTILS ###

def import_data(dir):
    return read_csv(dir)

### SOLUTIONS ###

def q1b():
    # Import data and take first 500 rows
    data = import_data(Q1_DATA_DIR)
    data = data.head(500)

    # Create folds
    folds = []
    fold_size = 50
    for i in range(0,10):
        folds.append(data[i*fold_size:i*fold_size+fold_size])

    # Create grid of 100 C values
    C_grid = linspace(0.0001, 0.6, 100)

    # Iterate over C values and train Logistic Model
    scores = []
    for C in C_grid.tolist():
        # Create model
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C)

        # Iterate over each fold to train and test on each
        fold_scores = []
        for i in range(0, len(folds)):
            # Set up training and test data
            folds_train = folds.copy()  # Create a temp so that we don't mutate original 'folds'
            test_df = folds_train.pop(i)
            train_df = concat(folds_train)

            # Fit model to data
            clf = model.fit(train_df.drop("Y", axis=1), train_df["Y"])

            # Score data
            clf_probs = clf.predict_proba(test_df.drop("Y", axis=1))
            fold_score = log_loss(test_df["Y"], clf_probs)
            fold_scores.append(fold_score)

        scores.append(fold_scores)

    # Save boxplot
    fig, ax = plt.subplots()
    ax.boxplot(scores)
    ax.set_xticklabels("")
    ax.set_title("Logistic Regression Log-Loss For Various C Values")
    ax.set_xlabel("C Values")
    ax.set_ylabel("Log-Loss")
    plt.savefig("./outputs/q1b_boxplot.png")

    # Record the value of C that returns the "best" CV score
    #
    # "Best" is taken to be the lowest average log-loss
    averages = list(map(lambda x : mean(x), scores))

    # Get index of lowest average
    i = averages.index(min(averages))

    # Map this to a C value
    C_best = C_grid[i]
    print(C_best)

    # Retrain this model and report its accuracy
    model = LogisticRegression(penalty="l1", solver="liblinear", C=C_best)

    # Let's test on the first fold and train on the remainder
    test_df = folds.pop(0)
    train_df = concat(folds)
    clf = model.fit(train_df.drop("Y", axis=1), train_df["Y"])

    print(clf.score(test_df.drop("Y", axis=1), test_df["Y"]))


def q1c():
    # Import data
    data = import_data(Q1_DATA_DIR)
    data = data.head(500)
    Xtrain = data.drop("Y", axis=1)
    Ytrain = data["Y"]

    # Create grid of 100 C values
    C_grid = linspace(0.0001, 0.6, 100)
    param_grid = { "C": C_grid }

    # Set our own scoring metric
    scoring = make_scorer(
        log_loss, 
        greater_is_better=False, 
        needs_proba=True
    )

    # Assignment code
    grid_lr = GridSearchCV(
        estimator=LogisticRegression(penalty='l1', solver='liblinear'),
        cv=10,
        param_grid=param_grid,
        scoring=scoring
    )
    grid_lr.fit(Xtrain, Ytrain)
    print(grid_lr.best_estimator_)
    print(grid_lr.best_score_)

### MAIN ###

def command_line_parsing():
    parser = argparse.ArgumentParser(description="Main script for the classifier model")
    parser.add_argument(
        "--question", metavar="question", type=str, help=f"Tell the script which question to run"
    )

    return parser

if __name__ == "__main__":
    args = command_line_parsing().parse_args()

    questions = {
        "q1b": q1b,
        "q1c": q1c
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
