# piperaceae-identification-paper

# Extractor
| Extract                           | Parameters          | Dimensions |
|-----------------------------------|---------------------|------------|
| Speed Up Robust Features (SURF)   | Surf Size=64        | 257        |
| Local Binary Pattern (LBP)        | P=8 and R=2 11-8bit | 256        |
| VGG16[^*]                         |                     | 512        |
| ResNet50[^*]                      |                     | 2048       |
| MobileNet-V2[^*]                  |                     | 1280       |

[^*]: ImageNet weights

# Classifier
| Classifiers                  | Parameters         | Values   |
|------------------------------|--------------------|----------|
| Decision Tree (DT)           | max_depth          | 10       |
| k-Nearest Neighboor (k-NN)   | n_neighboors       | 10       |
|                              | weights            | distance |
| Multilayer Perceptron (MLP)  | activation         | logistic |
|                              | learning_rate_init | 0.01     |
|                              | momentum           | 0.4      |
| Random Forest (RF)           | n_estimators       | 800      |
|                              | max_depth          | 100      |
| Support Vector Machine (SVM) | kernel             | rbf      |

# Dataset
- Images from [speciesLink](https://specieslink.net/)
- 375 images in grayscale (256x256)
- 375 images of Piperaceae family 
    - 75 of Manekia genus (f1)
    - 75 of Ottonia genus (f2)
    - 75 of Peperomia genus (f3)
    - 75 of Piper genus (f4)
    - 75 of Pothomorphe genus (f5)
- [Download](https://bit.ly/3OKxJK0)


# Experiments
- Patches: 3, 5, 7 (horizontal, vertical and horizontal+vertical)
- [Normalize](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- PCA: 128, 256, 512, 1024
- Fold: 5
- Train size: 80% (300)
- Test size: 20% (75)

# Accuracies
|                              |              | Slices       |              |
|------------------------------|--------------|--------------|--------------|
| Classifiers                  | #3           | #5           | #7           |
| Decision Tree (DT)           | 55.73 ±6.22  | 49.87 ±7.23  | 50.67 ±4.77  |
| k-Nearest Neighboor (k-NN)   | 76.27 ±1.96  | 74.67 ±2.39  | 67.73 ±5.56  |
| Multilayer Perceptron (MLP)  | 79.20 ±2.32  | 76.00 ±2.39  | 78.67 ±2.92  |
| Random Forest (RF)           | 77.87 ±2.87  | 77.60 ±1.77  | 73.87 ±3.44  |
| Support Vector Machine (SVM) | 80.53 ±3.64  | 78.67 ±1.89  | 75.47 ±4.18  |
- [Click on the link for more details](https://bit.ly/3PMPKsB)

# How to cite
```
@article{KAJIHARA2022,
    author = {Kajihara, Alexandre Y. and Bertolini, Diego and Schwerz, André L.},
    title  = {Identification of herbarium specimens: a case study with Piperaceae Giseke family},
    conference = {International Conference on Systems Signals and Image Processing (IWSSIP)}
    edition = {29}
    month = {jun}
    note = {Paper  accepted for presentation.},
    year = {2022},
    url = {http://iwssip.bg/},
    urlaccessdate = {30 abr. 2022},
}
```

