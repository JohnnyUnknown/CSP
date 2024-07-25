# CSP (coordinate search program)
import cv2 as cv
import determ_coord as dt
from decimal import *
import math
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
        if m.distance < 0.45 * n.distance:
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
        # Добавление в список КТ главного изображения, совпадающих с КТ искомого
        matches.append(large_image_KP)
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
# def determ_coordinates(img, center):
#     y, x = img.shape
#     first_x = 46.163400
#     first_y = 48.245472
#     second_x = 46.169534
#     second_y = 48.238911
#     center_coord = [0, 0]
#
#     meters_dolong = round((6378137 * math.cos(float(first_y)*math.pi/180) * 2 * math.pi) / 360, 3)
#     meters_dolat = 111100
#
#     pixel_price_x = (second_x - first_x) / x if first_x < second_x else (first_x - second_x) / x
#     pixel_price_y = (second_y - first_y) / y if first_y < second_y else (first_y - second_y) / y
#
#     center_coord[0] = round((pixel_price_x*center[0]+first_x), 6) if first_x < second_x else round(first_x-(pixel_price_x*center[0]), 6)
#     center_coord[1] = round((pixel_price_y*center[1]+first_y), 6) if first_y < second_y else round(first_y-(pixel_price_y*center[1]), 6)
#     print(*center_coord)

# Изменение размера изображения с сохранением пропорций
def resize_img(img, new_width):
    new_height = int(img.shape[0] * (new_width / img.shape[1]))
    # Изменение размера изображения с сохранением пропорций
    resized_image = cv.resize(img, (new_width, new_height))
    return resized_image

# Маска проверки найденных КТ на карте
def pixel_mask(matches):    # принимаются координаты КТ главного изображения
    # correct_matches = []
    # sum_x = 0
    # sum_y = 0
    # mask_correction = 2
    # for i in range(len(matches)):
    #     sum_x += matches[i][0]
    #     sum_y += matches[i][1]
    # medium_x = int(sum_x / len(matches))
    # medium_y = int(sum_y / len(matches))
    # # Нахождение коэффициента разницы высот полета и главного снимка для маски
    # height_coefficient = int(height_map / flight_altitude)
    #
    # for i in range(len(matches)):
    #     if ((matches[i][1] > medium_y - img1.shape[0] / (height_coefficient*mask_correction))
    #             and (matches[i][1] < medium_y + img1.shape[0] / (height_coefficient*mask_correction))):
    #         if ((matches[i][0] > medium_x - img1.shape[1] / (height_coefficient*mask_correction))
    #                 and (matches[i][0] < medium_x + img1.shape[1] / (height_coefficient*mask_correction))):
    #             correct_matches.append(matches[i])
    # return correct_matches

    correct_matches = []
    mask_correction = 2
    match_x = sorted(matches, key=lambda i: i[1])
    match_y = sorted(matches)

    if len(matches) % 2 == 0:
        indx1 = int(len(matches) / 2 - 1)
        indx2 = int(len(matches) / 2)
        median_x = (match_x[indx1][1] + match_x[indx2][1]) / 2
        median_y = (match_y[indx1][0] + match_y[indx2][0]) / 2
    else:
        indx = int((len(matches) - 1) / 2)
        median_x = match_x[indx][1]
        median_y = match_y[indx][0]

    # Нахождение коэффициента разницы высот полета и главного снимка для маски
    height_coefficient = int(height_map / flight_altitude)

    for i in range(len(matches)):
        if ((matches[i][0] >= median_y - img1.shape[1] / (height_coefficient * mask_correction))
                and (matches[i][0] < median_y + img1.shape[1] / (height_coefficient * mask_correction))):
            if ((matches[i][1] >= median_x - img1.shape[0] / (height_coefficient * mask_correction))
                    and (matches[i][1] < median_x + img1.shape[0] / (height_coefficient * mask_correction))):
                correct_matches.append(matches[i])
    return correct_matches

def stitcher(images_list):
    # Создание объекта для сшивания изображений
    stitcher = cv.Stitcher_create()
    # Сшивка изображений
    result = stitcher.stitch(images_list)
    print(result[0])
    # Проверка на успешность сшивки
    if result[0] == 0:  # Успешное сшивание
        # Отображение сшитого изображения
        cv.imshow("Stitched Image", result[1])
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Не удалось выполнить сшивание изображений.")

# image1 = cv.imread("C:\\My\\Projects\\images\\main\\1-1.jpg")
# image2 = cv.imread("C:\\My\\Projects\\images\\main\\1-2.jpg")
# image3 = cv.imread("C:\\My\\Projects\\images\\main\\2-1.jpg")
# image4 = cv.imread("C:\\My\\Projects\\images\\main\\2-2.jpg")
#
# stitcher([image1, image2, image3, image4])




# Главная карта

# # W_00002-1
# first_x = 46.162187
# first_y = 48.243968
# second_x = 46.163688
# second_y = 48.241215
# f_x = 46.161253
# f_y = 48.240687
# WK-00005
first_x = 46.164128
first_y = 48.246039
second_x = 46.166395
second_y = 48.238964
f_x = 46.159946
f_y = 48.237859
p1 = (first_y, first_x)
p2 = (second_y, second_x)
p3 = (f_y, f_x)

path_main = 'C:\\My\\Projects\\images\\main\\WK_00005.jpg'
img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)
img1 = cv.GaussianBlur(img1, (5, 5), sigmaX=0, sigmaY=0)
obj = dt.Determ_coord(p1, p2, p3, img1.shape)
# img1 = resize_img(img1, 3840)
kp1, des1 = search_KP(img1)
height_map = 500        # Высота съемки карты

# Видео с дрона
path_video = 'C:\\My\\Projects\\images\\move3.mp4'
cap = cv.VideoCapture(path_video)

frame_count = 0     # int(cap.get(cv.CAP_PROP_FPS))
f_cnt = 0
coord_cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Конец видеофайла.")
        break

    frame_count += 1
    if frame_count == 25:
        f_cnt += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = resize_img(gray, 1024)
        gray = cv.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        flight_altitude = 30        # текущая высота полета
        cv.imshow(" ", gray)
        cv.waitKey(0)
        cv.destroyAllWindows()

        kp2, des2 = search_KP(gray)
        frame_count = 0
        good_matches = matcher(des1, des2)
        if good_matches != None:
            # поиск общих КТ на главном изображении
            main_matches = location_images(good_matches, kp1)
            # Сравнение найденных общих КТ с маской проверки
            main_matches = pixel_mask(main_matches)
            if len(main_matches) > 1:
                coord_cnt += 1
                center = search_center(main_matches)
                print(len(main_matches), center)
                print_map(center)
                print(obj.calculate(center))
            else:
                print("Совпадений не найдено.")

        # # показывает полет наглядно, но тормозит программу
        # plt.imshow(img1, "gray"), plt.show(block=False)
        # plt.pause(0.01)
print(f"Всего кадров: {f_cnt}; найдено: {coord_cnt}")

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()