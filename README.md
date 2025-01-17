# piperaceae-identification-paper

These two other repositories are related:
- [piperaceae-segmentation](): contains the code used to remove background in the exsiccate.
- [piperaceae-features](): contains the code used to extract features from images.

---

## How to run

```
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
$ python main.py [ARGS]
```

List of args available:

- `-C, --config` [**NO REQUIRED**]: path to the configuration file.
  - - In the next version.

- `-c, --clf` [**NO REQUIRED**]: classifier to run experiments.
  - The classifiers available are: DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, MLPClassifier, SVC.
  - DEFAULT: `DecisionTreeClassifier`;

- `-i, --input` [**REQUIRED**]: path to features.
  - DEFAULT is `DecisionTreeClassifier`.
  - The `--input` must contain files in formats {`.npy`, `.npz`, `.csv` or `.txt`} and the other two files (`dataset.csv` and `samples.csv`).
    - `dataset.csv` and `samples.csv` are generated by this [code](). More details are available in this [repository]().

- `-o, --output` [**NO REQUIRED**]: path to save.
  - DEFAULT: create a folder named output;

- `-p, --pca` [**NO REQUIRED**]: enable or no PCA
  - In the next version.

---

## Source code:
- `figures`: contains a code that converts Confusion Matrix (DataFrame) to image.
- `sql`: load the best results to a database;
- `arrays.pyx`,  `setup.py`: used to Cython;
  - Cython is used to load the dataset faster.
- `classifiers.py`: functions that select the classifiers;
- `config.py`: had configurations for this experiment (fold, seed, etc.);
- `dataset.py`: load the information of the dataset and the own dataset;
- `df.py`: functions that transform the outputs into a data frame;
- `evaluate.py`: functions to evaluate `y_pred` and `y_true`;
- `fold.py`: a class that finds the best hyperparameters and executes the folds.
- `features.py`: class that contains the size of features;
- `image.py`: class to store image information used;
- `level.py`: class used to save levels used in experiments;
  - This class is used to generate a Confusion Matrix by class;
- `main.py`: main file that invokes other files;
- `mean.py`: class that calculated the means;
- `result.py`: class that saves the results;
- `sample.py`: class to store image information used;
- `save.py`: functions used to save outputs.

---

## Output

This code creates folders based on the number of folds and two folders (`mean` and `best`):

- `mean`: contains CSV files with mean and standard deviation using F1-Score, Accuracy, Top-k, and count of True Positive;
  - The True Positive is calculated using the Confusion Matrix.
- `best`: had CSV files with the highest rates achieved in this execution;
- `fold+1`...`fold+n`: the number folder called fold depends on the number fold set to run. In this folder was available information about:
  - `confusion_matrix`
    - `normalized`: is a Confusion Matrix with all classes and values between 0 and 1;
    - `multilabel`: is a Confusion Matrix for each class.
  - `best_evals.csv`: the rule that achieved the highest rate on F1-Score and accuracy;
  - `evals.csv`:  the F1-Score and accuracy for each rule;
  - `infos.csv`: time to train and count of samples used in train and test;
  - `preds.csv`: list of predictions generated by the model;
  - `topk.csv`: count of hits to k = 3 to k = n-1, where n is a count of classes;
  - `true_positive.csv`: count of true positives for each class.

In each zip available are the exsiccata used in three dimensions (256 x 256, 400 x 400, 512 x 512), with three folders for each size.
jpeg: the exsiccate resized.
mask: the mask predicted for U-Net.
segmented: image without background.