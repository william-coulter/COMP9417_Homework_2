import argparse

from numpy import linspace
from pandas import read_csv, concat
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

DATA_DIR = "./data/Q1.csv"

### UTILS ###

def import_data():
    return read_csv(DATA_DIR)

### SOLUTIONS ###

def q1b():
    # Import data and take first 500 rows
    data = import_data()
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
            fold_score = clf.score(test_df.drop("Y", axis=1), test_df["Y"])
            fold_scores.append(fold_score)

        scores.append(fold_scores)
    
    # Save boxplot
    fix, ax = plt.subplots()
    ax.boxplot(scores)
    ax.set_xticklabels(C_grid)
    ax.set_title("C Values")
    plt.savefig("./boxplot.png")


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
        "q1b": q1b
    }

    # Execute the given command
    question = questions.get(args.question)
    question()
