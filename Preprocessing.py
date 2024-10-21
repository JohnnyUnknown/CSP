import cv2 as cv
import numpy as np


# Методы предобработки изображений
def clahe_improvement(img):
    clahe = cv.createCLAHE(2, (5, 5))
    # bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    # l, a, b = cv.split(lab)
    # l2 = clahe.apply(l)
    # lab = cv.merge((l2, a, b))
    # img2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    # img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img2 = clahe.apply(img)
    return img2


def gauss_improvement(img):
    # Дилатация (увеличение светлых пятен)
    # img2 = cv.dilate(img, (3, 3), iterations=1)

    # Эрозия (уменьшение светлых пятен)
    # img2 = cv.erode(img, (3, 3), iterations=1)

    img2 = cv.GaussianBlur(img, (5, 5), sigmaX=0, sigmaY=0)

    # Медианное размытие (при "царапинах" на изображении)
    # img2 = cv.medianBlur(img, 5)

    # Повышение резкости изображения
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Фильтр Собеля (обозначает контуры)
    # kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    # Фильтр лапласиан (более качественно обозначает контуры)
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # img2 = cv.filter2D(img, -1, kernel)
    return img2


def resize_img(img, new_width):
    new_height = int(img.shape[0] * (new_width / img.shape[1]))
    # Изменение размера изображения с сохранением пропорций
    resized_image = cv.resize(img, (new_width, new_height))
    return resized_image


def definition_of_blur(height, altitude):
    diff = int(height / altitude)
    if diff <= 5:
        return 5, 5
    elif diff % 2 == 1:
        return diff, diff
    else:
        return diff + 1, diff + 1
