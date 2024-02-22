import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras import regularizers
import random

from PyQt5.QtWidgets import *
from f74076108_Hw1_Q5 import *
import f74076108_Hw1_Q5 as ui

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.showTrainImage)
        self.pushButton_2.clicked.connect(self.showHyper)
        self.pushButton_3.clicked.connect(self.showModelShortcut)
        self.pushButton_4.clicked.connect(self.showACCuracy_Loss)
        self.pushButton_5.clicked.connect(self.modelTest)

        self.label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
    
    def showTrainImage(self):
        plt.figure(figsize=(9, 9))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            j = random.randint(0, 49999)
            plt.title(self.label[int(self.y_train[j][0])])
            # print(y_train[j].dtype)
            plt.axis('off')
            plt.imshow(self.x_train[j])
        plt.show()
    
    def showHyper(self):
        batch_size = 128
        learning_rate = 0.1
        optimizer = 'SGD'
        print('hyperparameters:')
        print('batch size: ', format(batch_size))
        print('learning rate: ', format(learning_rate))
        print('optimizer: ', format(optimizer))

    def showModelShortcut(self):
        with open('model_arch.json', 'r') as f:
            json_string = f.read()
        model = Sequential()
        model = model_from_json(json_string)
        print(model.summary())

    def showACCuracy_Loss(self):
        plt.figure(figsize=(9, 9))
        plt.subplot(2, 1, 1)
        acc = plt.imread('model_accuracy.png')
        plt.imshow(acc)
        plt.subplot(2, 1, 2)
        loss = plt.imread('model_loss.png')
        plt.imshow(loss)
        plt.show()

    def modelTest(self):
        msg = self.lineEdit.text()
        with open('model_arch.json', 'r') as f:
            json_string = f.read()
        model = Sequential()
        model = model_from_json(json_string)
        # print(model.summary())
        model.load_weights('cifar10_vgg16.h5', by_name=False)
        # print(msg)
        x_test = self.x_test.astype(np.float32)
        x = x_test[int(msg)]
        # print(x.shape)
        # print(np.array([x]))
        predict = model.predict(np.array([x]))
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.x_test[int(msg)])
        plt.subplot(1, 2, 2)
        plt.bar(self.label, predict[0])
        plt.xticks(rotation=45)
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = Main()
    mainwindow.show()
    sys.exit(app.exec_())
