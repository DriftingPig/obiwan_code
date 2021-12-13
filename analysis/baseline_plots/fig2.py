#confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import matplotlib.mlab as mlab
import matplotlib

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = None

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    #obiwan
    cm = cm.transpose()
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title,
           ylabel='measure',
           xlabel='Truth')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

def fig2(catalog,startid):
    source = catalog.processed_one
    sel = source['matched']
    source = source[sel]
    y_true = np.zeros(len(source))
    y_true[source['sim_sersic_n']==0] = 0
    y_true[source['sim_sersic_n']==1] = 2
    y_true[(source['sim_sersic_n']==1)&(source['sim_e1']==0)&(source['sim_e2']==0)] = 1
    y_true[source['sim_sersic_n']==4] = 3
    y_true[(source['sim_sersic_n']!=4)&(source['sim_sersic_n']!=1)&(source['sim_sersic_n']!=0)] = 4
    y_pred = np.zeros(len(source))
    y_pred[source['type']=='PSF']=0
    y_pred[source['type']=='REX']=1
    y_pred[source['type']=='EXP']=2
    y_pred[source['type']=='DEV']=3
    y_pred[source['type']=='SER']=4
    y_true = np.array(y_true,dtype = np.int)
    y_pred = np.array(y_pred,dtype = np.int)
    class_names = np.array(['PSF','REX', 'EXP','DEV','SER'],dtype=np.str)
    plot_confusion_matrix(y_true, y_pred, classes=class_names,normalize=True,
                      title=None)
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    plt.savefig(topdir+'/fig2.png') 