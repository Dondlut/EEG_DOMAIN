# File Description

## data

- *patient_1. py*  *patient_2. py* and *patient_3. py* are assigned patients for the training set, validation set, and testing set, respectively. Due to the possibility of severe class imbalance during the random allocation process, the distribution of some patients in the three subsets was manually adjusted.

- *train_patient.txt*  *dev_patient.txt*  and  *test_patient.txt* display the patients in the training set, validation set, and test set respectively. The format is  *Patient ID: {Epilepsy Category: Number of Epilepsy Samples}*

## visualize

- ### fig

  - Visualization results, folder name represents epilepsy category

- ### h5_anno

  - Visualize samples, folder name represents epilepsy category

- ### model

  - Trained models

- temp  Visualize code

*confusion_matrix.py* is used to generate confusion matrices

*constant_22.py*  is used to obtain patient IDs

*train_c.py* is the training code
