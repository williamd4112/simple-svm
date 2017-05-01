import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(func, x_, t_, xo_, to_, xt_, tt_, h):
    x_min = x_[:, 0].min()
    x_max = x_[:, 0].max()
    y_min = x_[:, 1].min()
    y_max = x_[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #Z = func(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(xt_[:, 0], xt_[:, 1], s=4, c=tt_, edgecolors='k', cmap=plt.cm.Paired)
    plt.scatter(x_[:, 0], x_[:, 1], s=50, c=t_, marker='o', cmap=plt.cm.Paired)
    plt.scatter(xo_[:, 0], xo_[:, 1], s=250, c=to_, marker='x', cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

if __name__ == '__main__':
    plot_decision_boundary(None, 0, 0, 255, 255, 10)
