# CSP (coordinate search program)
import cv2 as cv
from decimal import *
import numpy as np
import matplotlib.pyplot as plt

# Поиск КТ изображения
def search_KP(img):
    # Инициализация метода SIFT
    sift = cv.SIFT_create()
    # Поиск КТ и их дескрипторов
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

# Поиск общих КТ двух изображений
def matcher(des1, des2):
    # Инициализация BFMatcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Нахождение общих точек
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)

    if len(good) >= 5:
        # Возврат списка общих КТ и кортежа КТ главного изображения
        return good
    else:
        print("Совпадений не найдено.")
        return None

# Поиск списка координат общих КТ на главном изображении
def location_images(good_matches, kp1):
    matches = []
    for i in range(len(good_matches)):
        dmatch = good_matches[i]
        # Поиск найденных КТ для обеих изображений в списке КТ главного изображения
        large_image_KP = list(kp1[dmatch.queryIdx].pt)
        large_image_KP[0] = int(large_image_KP[0])
        large_image_KP[1] = int(large_image_KP[1])
        # print(large_image_KP)
        # Добавление в список КТ главного изображения, совпадающих с КТ искомого
        matches.append(large_image_KP)
        # print(matches[i])
    return matches

# Поиск прямоугольника, образующего искомую область на главном изображении
def search_center(matches):
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
    thickness = 20
    img3 = cv.rectangle(img1, start_point, end_point, color, thickness)
    cv.imwrite("Points.jpg", img3)

# Определение географических координат местоположения дрона
def determ_coordinates(img, center):
    y, x = img.shape
    # MapFree cam5
    first_x = Decimal(46.146381)
    first_y = Decimal(48.248815)
    second_x = Decimal(46.171939)
    second_y = Decimal(48.237338)

    # MapZoom
    # first_x = Decimal(46.159611)
    # first_y = Decimal(48.240549)
    # second_x = Decimal(46.162097)
    # second_y = Decimal(48.245180)

    #MapFree2
    # first_x = Decimal(46.153183)
    # first_y = Decimal(48.260676)
    # second_x = Decimal(46.203937)
    # second_y = Decimal(48.245071)

    # MapBolhun
    # first_x = Decimal(46.443410)
    # first_y = Decimal(48.046189)
    # second_x = Decimal(46.595268)
    # second_y = Decimal(48.002060)

    # MapBolhun2
    # first_x = Decimal(46.466746)
    # first_y = Decimal(48.042403)
    # second_x = Decimal(46.545223)
    # second_y = Decimal(48.019823)

    pixel_price_x = Decimal((second_x - first_x) / x)
    pixel_price_y = Decimal((first_y - second_y) / y)
    center_coord = [round((pixel_price_x*center[0]+first_x),6), round(first_y-(pixel_price_y*center[1]),6)]
    print(*center_coord)

# Изменение размера изображения с сохранением пропорций
def resize_img(img, new_width):
    new_height = int(img.shape[0] * (new_width / img.shape[1]))
    # Изменение размера изображения с сохранением пропорций
    resized_image = cv.resize(img, (new_width, new_height))
    return resized_image

# Маска проверки найденных КТ на карте
def pixel_mask(matches, height_coefficient):    # принимаются координаты КТ главного изображения
    correct_matches = []
    sum_x = 0
    sum_y = 0
    for i in range(len(matches)):
        sum_x += matches[i][0]
        sum_y += matches[i][1]
    medium_x = int(sum_x / len(matches))
    medium_y = int(sum_y / len(matches))

    for i in range(len(matches)):
        if (matches[i][1] > medium_y - img1.shape[1] / height_coefficient) and (matches[i][1] < medium_y + img1.shape[1] / height_coefficient):
            if (matches[i][0] > medium_x - img1.shape[1] / height_coefficient) and (matches[i][0] < medium_x + img1.shape[1] / height_coefficient):
                correct_matches.append(matches[i])
    # print(correct_matches)
    return correct_matches

# Нахождение коэффициента разницы высот полета и главного снимка для маски
def find_height_coefficient(height_map, flight_altitude):
    height_coef = int(height_map/flight_altitude)   # доп. умножение на 1.2 показывает более точные результаты
    # print(height_coef)
    return height_coef


# Главная карта
path_main = 'C:\\My\\Projects\\images\\main\\WK_00005-12.jpg'
img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)
kp1, des1 = search_KP(img1)
# Высота съемки карты
height_map = 500

# Видео с дрона
path_video = 'C:\\My\\Projects\\images\\move3.mp4'
cap = cv.VideoCapture(path_video)

frame_count = 0     # int(cap.get(cv.CAP_PROP_FPS))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Конец видеофайла.")
        break

    frame_count += 1
    if frame_count == 25:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = resize_img(gray, 1280)
        # текущая высота полета
        flight_altitude = 30

        kp2, des2 = search_KP(gray)
        frame_count = 0
        good_matches = matcher(des1, des2)
        if good_matches != None:
            # поиск общих КТ на главном изображении
            main_matches = location_images(good_matches, kp1)
            # Сравнение найденных общих КТ с маской проверки
            main_matches = pixel_mask(main_matches, find_height_coefficient(height_map, flight_altitude))
            if len(main_matches) > 2:
                center = search_center(main_matches)
                print_map(center)
                determ_coordinates(img1, center)
            else:
                print("Совпадений не найдено.")

        # показывает полет наглядно, но тормозит программу
        # plt.imshow(img1, "gray"), plt.show(block=False)
        # plt.pause(0.01)


cv.waitKey(0)
cap.release()
cv.destroyAllWindows()