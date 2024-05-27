# CSP (coordinate search program)
import cv2 as cv
from decimal import *
import numpy as np
import matplotlib.pyplot as plt

# Поиск КТ двух изображений и общих КТ
def search_KP(img1, img2):
    # Инициализация метода SIFT
    sift = cv.SIFT_create()

    # Поиск КТ и их дескрипторов
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Инициализация BFMatcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Нахождение общих точек
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    if len(good) >= 5:
        # Возврат списка общих КТ и кортежа КТ главного изображения
        return good, kp1
    else:
        print("Совпадений не найдено.")
        return None, None

# Поиск списка координат общих КТ на главном изображении
def location_images(good_matches, kp1):
    matches = []
    for i in range(len(good_matches)):
        dmatch = good_matches[i]
        # Поиск найденных КТ для обеих изображений в списке КТ главного изображения
        large_image_KP = list(kp1[dmatch.queryIdx].pt)
        large_image_KP[0] = int(large_image_KP[0])
        large_image_KP[1] = int(large_image_KP[1])
        # Добавление в список КТ главного изображения, совпадающих с КТ искомого
        matches.append(large_image_KP)
    return matches

# Поиск прямоугольника, образующего искомую область на главном изображении
def search_center(img1, matches):
    list_x = []
    list_y = []
    for i in range(len(matches)):
        list_x.append(matches[i][0])
        list_y.append(matches[i][1])
    list_x.sort()
    list_y.sort()
    # Нахождение центральной точки искомого изображения на главном изображении
    center_x = int((list_x[0] + list_x[-1]) / 2)
    center_y = int((list_y[0] + list_y[-1]) / 2)
    return [center_x, center_y]

# Отображение местоположения дрона на главном изображении
def print_map(center):
    start_point = center
    end_point = center
    color = (0, 0, 255)
    thickness = 5
    img3 = cv.rectangle(img1, start_point, end_point, color, thickness)
    cv.imshow(" ", img3)
    # cv.waitKey(0)

# Определение географических координат местоположения дрона
def determ_coordinates(img, center):
    y, x, _ = img.shape
    # first_x = Decimal(46.146381)
    # first_y = Decimal(48.248815)
    # second_x = Decimal(46.171939)
    # second_y = Decimal(48.237338)

    first_x = Decimal(46.153183)
    first_y = Decimal(48.260676)
    second_x = Decimal(46.203937)
    second_y = Decimal(48.245071)

    pixel_price_x = Decimal((second_x - first_x) / x)
    pixel_price_y = Decimal((first_y - second_y) / y)
    center_coord = [round((pixel_price_x*center[0]+first_x),6), round(first_y-(pixel_price_y*center[1]),6)]
    print(*center_coord)


# Главная карта
path_main = 'C:\\My\\Projects\\images\\MapFree2.png'
img1 = cv.imread(path_main)
image_main = cv.imread(path_main, cv.IMREAD_GRAYSCALE)

# Снимок с дрона
# path_photo = 'C:\\My\\Projects\\images\\photo1.png'
# photo = cv.imread(path_photo, cv.IMREAD_GRAYSCALE)

# Видео с дрона
path_video = 'C:\\My\\Projects\\images\\cam3.mp4'
cap = cv.VideoCapture(path_video)

frame_count = 0     # int(cap.get(cv.CAP_PROP_FPS))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось открыть видео.")
        break

    frame_count += 1
    if frame_count == 30:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_count = 0
        good_matches, keypoints1 = search_KP(image_main, gray)
        if good_matches != None:
            main_matches = location_images(good_matches, keypoints1)
            center = search_center(image_main, main_matches)
            print_map(center)
            determ_coordinates(img1, center)

        # cv.imshow("frame", gray)
        if cv.waitKey(3) & 0xFF == ord('z'):
            break

cap.release()
cv.destroyAllWindows()