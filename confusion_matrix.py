import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

label_dict = {0: 'CFSZ', 1: 'GNSZ', 2: "ABSZ", 3: "CTSZ"}

def plot_confusion_matrix(y_true, y_pred, title="cross_val",
                          cmap=plt.cm.Blues, save_flg=True):
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    classes = [label_dict[i] for i in range(4)]
    labels = range(4)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    # print('Confusion matrix, without normalization')

    # thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center", fontsize=30)

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    if save_flg:
        plt.savefig(r"./{}.png".format(title))
    plt.close()
