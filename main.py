import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from numpy.core.records import array
from scipy import signal
from f74076108_hw1 import *
import f74076108_hw1 as ui

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.colorSeparation)
        self.pushButton_3.clicked.connect(self.colorTransformation)
        self.pushButton_4.clicked.connect(self.blending)
        self.pushButton_5.clicked.connect(self.gaussianBlur)
        self.pushButton_6.clicked.connect(self.bilateralFilter)
        self.pushButton_7.clicked.connect(self.medianFilter)
        self.pushButton_8.clicked.connect(self.gaussianGeneration)
        self.pushButton_9.clicked.connect(self.sobelX)
        self.pushButton_10.clicked.connect(self.sobelY)
        self.pushButton_11.clicked.connect(self.magnitude)
        self.pushButton_12.clicked.connect(self.imgResize)
        self.pushButton_13.clicked.connect(self.imgTranslation)
        self.pushButton_14.clicked.connect(self.imgRotate_Scale)
        self.pushButton_15.clicked.connect(self.imgShearing)

    def loadImage(self):
        img = cv2.imread('Q1_Image\Sun.jpg')
        print("Height :", img.shape[0])
        print("Width :", img.shape[1])
        cv2.imshow('Hw1-1', img)

    def colorSeparation(self):
        img = cv2.imread('Q1_Image\Sun.jpg')
        (B, G, R) = cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype = np.uint8)

        cv2.imshow("B channel", cv2.merge([B, zeros, zeros]))
        cv2.imshow("G channel", cv2.merge([zeros, G, zeros]))
        cv2.imshow("R channel", cv2.merge([zeros, zeros, R]))

    def colorTransformation(self):
        img = cv2.imread('Q1_Image\Sun.jpg')
        (B, G, R) = cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype = np.uint32)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = zeros + (B / 3 + G / 3 + R / 3)
        img2 = img2.astype('uint8')
        cv2.imshow("I1", img1)
        cv2.imshow("I2", img2)

    def blending(self):
        img1 = cv2.imread('Q1_Image\Dog_Strong.jpg')
        img2 = cv2.imread('Q1_Image\Dog_Weak.jpg')
        alpha = 0.5
        beta = 1 - alpha
        def betaUpdate(x):
            global beta, alpha
            beta = x/256
            alpha = 1 - beta
            img3 = cv2.addWeighted(img1, alpha, img2, beta, 0)
            cv2.imshow("Blend", img3)
        
        cv2.namedWindow('Blend')
        cv2.createTrackbar("Blend:", "Blend", 0, 255, betaUpdate)
        cv2.setTrackbarPos("Blend:", "Blend", 125)

    def gaussianBlur(self):
        img = cv2.imread('Q2_Image\Lenna_whiteNoise.jpg')
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow("gaussian Blur", img)

    def bilateralFilter(self):
        img = cv2.imread('Q2_Image\Lenna_whiteNoise.jpg')
        img = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.imshow("Bilateral Filter", img)

    def medianFilter(self):
        img = cv2.imread('Q2_Image\Lenna_pepperSalt.jpg')
        img1 = cv2.medianBlur(img, 3)
        img2 = cv2.medianBlur(img, 5)
        cv2.imshow("Median Filter 3*3", img1)
        cv2.imshow("Median Filter 5*5", img2)

    def gaussianGeneration(self):
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2)) #sigma = 0.705
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # print(gaussian_kernel)

        img = cv2.imread('Q3_Image\House.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img3 = cv2.filter2D(img1, -1, gaussian_kernel)
        img2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        img1 = np.pad(array=img1, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        # print(img1)
        for i in range(1, img1.shape[0]-1):
            for j in range(1, img1.shape[1]-1):
                img2[i-1, j-1] = img1[i-1, j-1]*gaussian_kernel[0, 0] + img1[i, j-1]*gaussian_kernel[1, 0] + img1[i+1, j-1]*gaussian_kernel[2, 0] + \
                             img1[i-1, j]*gaussian_kernel[0, 1] + img1[i, j]*gaussian_kernel[1, 1] + img1[i+1, j]*gaussian_kernel[2, 1] + \
                             img1[i-1, j+1]*gaussian_kernel[0, 2] + img1[i, j+1]*gaussian_kernel[1, 2] + img1[i+1, j+1]*gaussian_kernel[2, 2]
        img2 = img2.astype('uint8')
        # print(img2)
        # print(img3)
        cv2.imshow("Grayscale", img1)
        cv2.imshow("Gaussian Blur", img2)
        # cv2.imshow("Gaussian", img3)

    def sobelX(self):
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2)) #sigma = 0.705
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        img = cv2.imread('Q3_Image\House.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        img1 = np.pad(array=img1, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        for i in range(1, img1.shape[0]-1):
            for j in range(1, img1.shape[1]-1):
                img2[i-1, j-1] = img1[i-1, j-1]*gaussian_kernel[0, 0] + img1[i, j-1]*gaussian_kernel[1, 0] + img1[i+1, j-1]*gaussian_kernel[2, 0] + \
                             img1[i-1, j]*gaussian_kernel[0, 1] + img1[i, j]*gaussian_kernel[1, 1] + img1[i+1, j]*gaussian_kernel[2, 1] + \
                             img1[i-1, j+1]*gaussian_kernel[0, 2] + img1[i, j+1]*gaussian_kernel[1, 2] + img1[i+1, j+1]*gaussian_kernel[2, 2]
        
        kernel_X = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        # img4 = cv2.filter2D(img2, -1, kernel_X)
        img3 = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.int16)
        img2 = np.pad(array=img2, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        # for i in range(1, img2.shape[0]-1):
        #     for j in range(1, img2.shape[1]-1):
        #         img3[i-1, j-1] = img2[i-1, j-1]*kernel_X[0, 0] + img2[i, j-1]*kernel_X[1, 0] + img2[i+1, j-1]*kernel_X[2, 0] + \
        #                      img2[i-1, j]*kernel_X[0, 1] + img2[i, j]*kernel_X[1, 1] + img2[i+1, j]*kernel_X[2, 1] + \
        #                      img2[i-1, j+1]*kernel_X[0, 2] + img2[i, j+1]*kernel_X[1, 2] + img2[i+1, j+1]*kernel_X[2, 2]
        # img3 = img3.astype('int8')
        img3 = np.zeros(img2.shape, dtype=np.int8)
        for i in range(1, img2.shape[0]-1):
            for j in range(1, img2.shape[1]-1):
                value = kernel_X*img2[(i-1):(i+2), (j-1):(j+2)]
                img3[i, j] = min(255, max(0, value.sum()))
        img3 = img3.astype('uint8')
        # print(img3)
        # print(img4)
        cv2.imshow("Sobel X", img3)
        # cv2.imshow("Sobel xxx", img4)
        
    def sobelY(self):
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2)) #sigma = 0.705
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        img = cv2.imread('Q3_Image\House.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        img1 = np.pad(array=img1, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        for i in range(1, img1.shape[0]-1):
            for j in range(1, img1.shape[1]-1):
                img2[i-1, j-1] = img1[i-1, j-1]*gaussian_kernel[0, 0] + img1[i, j-1]*gaussian_kernel[1, 0] + img1[i+1, j-1]*gaussian_kernel[2, 0] + \
                             img1[i-1, j]*gaussian_kernel[0, 1] + img1[i, j]*gaussian_kernel[1, 1] + img1[i+1, j]*gaussian_kernel[2, 1] + \
                             img1[i-1, j+1]*gaussian_kernel[0, 2] + img1[i, j+1]*gaussian_kernel[1, 2] + img1[i+1, j+1]*gaussian_kernel[2, 2]

        kernel_Y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        # img3 = img3.astype('int8')
        img3 = np.zeros(img2.shape, dtype=np.int8)
        for i in range(1, img2.shape[0]-1):
            for j in range(1, img2.shape[1]-1):
                value = kernel_Y*img2[(i-1):(i+2), (j-1):(j+2)]
                img3[i, j] = min(255, max(0, value.sum()))
        img3 = img3.astype('uint8')
        
        # img3 = cv2.filter2D(img2, -1, kernel_Y)
        cv2.imshow("Sobel Y", img3)

    def magnitude(self):
        img = cv2.imread('Q3_Image\House.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        img1 = np.pad(array=img1, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2)) #sigma = 0.705
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        for i in range(1, img1.shape[0]-1):
            for j in range(1, img1.shape[1]-1):
                img2[i-1, j-1] = img1[i-1, j-1]*gaussian_kernel[0, 0] + img1[i, j-1]*gaussian_kernel[1, 0] + img1[i+1, j-1]*gaussian_kernel[2, 0] + \
                             img1[i-1, j]*gaussian_kernel[0, 1] + img1[i, j]*gaussian_kernel[1, 1] + img1[i+1, j]*gaussian_kernel[2, 1] + \
                             img1[i-1, j+1]*gaussian_kernel[0, 2] + img1[i, j+1]*gaussian_kernel[1, 2] + img1[i+1, j+1]*gaussian_kernel[2, 2]
        
        kernel_X = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        kernel_Y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        
        img3 = np.zeros(img2.shape, dtype=np.int8)
        for i in range(1, img2.shape[0]-1):
            for j in range(1, img2.shape[1]-1):
                value = kernel_X*img2[(i-1):(i+2), (j-1):(j+2)]
                img3[i, j] = min(255, max(0, value.sum()))
        # 
        img4 = np.zeros(img2.shape, dtype=np.int8)
        for i in range(1, img2.shape[0]-1):
            for j in range(1, img2.shape[1]-1):
                value = kernel_Y*img2[(i-1):(i+2), (j-1):(j+2)]
                img4[i, j] = min(255, max(0, value.sum()))
        # 

        # img3 = img3.astype('uint')

        img3 = img3.astype('uint32')
        img4 = img4.astype('uint32')

        mag = np.zeros(img2.shape, dtype=np.int32)
        mag = np.sqrt(np.square(img3) + np.square(img4))
        print(mag)
        # mag *= 255.0 / mag.max()
        mag = mag.astype('uint8')
        # mag = np.array(mag, dtype='uint8')
        print(mag)

        img3 = img3.astype('uint8')
        img4 = img4.astype('uint8')

        cv2.imshow("x", img3)
        cv2.imshow("y", img4)
        cv2.imshow("Magnitude", mag)

    def imgResize(self):
        img = cv2.imread('Q4_Image\SQUARE-01.png')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow("img", img)

    def imgTranslation(self):
        img = cv2.imread('Q4_Image\SQUARE-01.png')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        T = np.float32([
            [1, 0, 0],
            [0, 1, 60]
        ])
        img = cv2.warpAffine(img, T, (400, 300))
        cv2.imshow("img_2", img)
        
    def imgRotate_Scale(self):
        img = cv2.imread('Q4_Image\SQUARE-01.png')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 10, 0.5)
        img = cv2.warpAffine(img, M, (400, 300))
        T = np.float32([
            [1, 0, 0],
            [0, 1, 60]
        ])
        img = cv2.warpAffine(img, T, (400, 300))
        cv2.imshow("img_3", img)

    def imgShearing(self):
        img = cv2.imread('Q4_Image\SQUARE-01.png')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        T = np.float32([
            [1, 0, 0],
            [0, 1, 60]
        ])
        img = cv2.warpAffine(img, T, (400, 300))
        M1 = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), 10, 0.5)
        img = cv2.warpAffine(img, M1, (400, 300))
        pos1 = np.float32([
            [50, 50],
            [200, 50],
            [50, 200]
        ])
        pos2 = np.float32([
            [10, 100],
            [200, 50],
            [100, 250]
        ])
        M2 = cv2.getAffineTransform(pos1, pos2)
        img = cv2.warpAffine(img, M2, (400, 300))
        cv2.imshow("img_4", img)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Main()
    mainWindow.show()
    sys.exit(app.exec_())