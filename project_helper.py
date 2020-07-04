# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:32:25 2020

@author: sriva
"""

from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    plt.imsave(fname='Uploads/{}.png'.format(i), arr=X_test[i])
