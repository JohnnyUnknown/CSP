# CSP (coordinate search program)
import cv2 as cv
import math
import determ_coord as dt
import Kalman_filter as kf
import Filter
import Affine_transform
import json
from decimal import *
import numpy as np
import matplotlib.pyplot as plt


# Поиск КТ изображения
def search_KP(img):
    # Инициализация метода SIFT
    sift = cv.SIFT_create()
    # Поиск КТ и их дескрипторов
    kp, des = sift.detectAndCompute(img, None)

    # kp, des = Affine_transform.apply_asift_transformation(img)

    # akaze = cv.AKAZE_create()
    # kp, des = akaze.detectAndCompute(img, None)

    # orb = cv.ORB_create(nfeatures=250000)
    # kp, des = orb.detectAndCompute(img, None)

    # # Инициализация SURF детектора
    # surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)
    # kp, des = surf.detectAndCompute(img, None)
    # print(len(kp))

    return kp, des


# Поиск общих КТ двух изображений
def matcher(des1, des2):
    # Инициализация BFMatcher
    bf = cv.BFMatcher()
    # bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Нахождение общих точек
    good = [m for m, n in matches if m.distance < 0.5 * n.distance]

    # # Сопоставление дескрипторов с использованием KNN для ORB
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # # Инициализация BFMatcher с использованием L2 нормы для SURF
    # bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    # # Поиск соответствий между дескрипторами
    # matches = bf.match(des1, des2)
    # # Сортировка матчей по их расстоянию (чем меньше расстояние, тем лучше матч)
    # matches = sorted(matches, key=lambda x: x.distance)
    # # Фильтрация "хороших" матчей
    # good = [m for m in matches if m.distance < 0.7 * matches[-1].distance]

    if len(good) >= 4:
        # Возврат списка общих КТ и кортежа КТ главного изображения
        return good
    else:
        print("Общих точек не найдено.")
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


# Поиск списка координат общих КТ на кадре после pixel_mask
def location_images_2(good_matches, kp, matches_index):
    matches = []
    for i in range(len(good_matches)):
        if i in matches_index:
            large_image_KP = list(kp[good_matches[i].trainIdx].pt)
            large_image_KP[0] = int(large_image_KP[0])
            large_image_KP[1] = int(large_image_KP[1])
            matches.append(large_image_KP)
    return matches


def search_center(main_matches, matches_2):
    # Массивы с точками соответствия
    pts1 = np.float32([m for m in matches_2]).reshape(-1, 1, 2)
    pts2 = np.float32([m for m in main_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
    return H


def true_center(img, main_matches, matches):
    for i in range(len((matches))):
        print_map(gray, matches[i], 3)

    true_center = np.array([[img.shape[1] / 2, img.shape[0] / 2]], dtype='float32').reshape(-1, 1, 2)
    # print(f"{true_center=}, {true_center.shape}")

    # print("In true center")
    # print(len(main_matches), main_matches)
    # print(len(matches), matches)
    H = search_center(main_matches, matches)
    # print(H)
    find_center = cv.perspectiveTransform(true_center, H)
    find = []
    find.append(int(find_center[0][0][0]))
    find.append(int(find_center[0][0][1]))
    print(f"{find=}")
    return find


# Отображение местоположения дрона на главном изображении
def print_map(img1, center, thick):
    color = (0, 0, 255)
    img3 = cv.circle(img1, center, 3, color, thick)
    cv.imwrite("Points.jpg", img3)


def print_map_2(gray, thick):
    color = (0, 0, 255)
    thickness = thick
    center2 = []
    center2.append(int(gray.shape[1] / 2))
    center2.append(int(gray.shape[0] / 2))
    img3 = cv.circle(gray, center2, 3, color, thickness)
    cv.imshow(" ", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Изменение размера изображения с сохранением пропорций
def resize_img(img, new_width):
    new_height = int(img.shape[0] * (new_width / img.shape[1]))
    # Изменение размера изображения с сохранением пропорций
    resized_image = cv.resize(img, (new_width, new_height))
    return resized_image


# Маска проверки найденных КТ на карте
def pixel_mask(matches):  # принимаются координаты КТ главного изображения
    correct_matches = []
    correct_matches_index = []
    mask_correction = 1
    match_x = sorted(matches)
    match_y = sorted(matches, key=lambda i: i[1])

    if len(matches) % 2 == 0:
        indx1 = int(len(matches) / 2 - 1)
        indx2 = int(len(matches) / 2)
        median_y = (match_y[indx1][1] + match_y[indx2][1]) / 2
        median_x = (match_x[indx1][0] + match_x[indx2][0]) / 2
    else:
        indx = int((len(matches) - 1) / 2)
        median_y = match_y[indx][1]
        median_x = match_x[indx][0]

    # Нахождение коэффициента разницы высот полета и главного снимка для маски
    height_coefficient = round(height_map / flight_altitude, 2)

    for i in range(len(matches)):
        if ((matches[i][0] >= median_x - img1.shape[1] / (height_coefficient / mask_correction))
                and (matches[i][0] <= median_x + img1.shape[1] / (height_coefficient / mask_correction))):
            if ((matches[i][1] >= median_y - img1.shape[1] / (height_coefficient / mask_correction))
                    and (matches[i][1] <= median_y + img1.shape[1] / (height_coefficient / mask_correction))):
                correct_matches.append(matches[i])
                correct_matches_index.append(i)
    print(f"Общих точек: до {len(matches)}, после фильтра {len(correct_matches)}")

    for i in range(len(correct_matches)):
        print_map(img1, correct_matches[i], 5)
    # point = [int(median_x - img1.shape[1] / (height_coefficient / mask_correction)), int(median_y + img1.shape[1] / (height_coefficient / mask_correction))]
    # point2 = [int(median_x + img1.shape[1] / (height_coefficient / mask_correction)), int(median_y - img1.shape[1] / (height_coefficient / mask_correction))]
    # point3 = [int(median_x - img1.shape[1] / (height_coefficient / mask_correction)), int(median_y - img1.shape[1] / (height_coefficient / mask_correction))]
    # point4 = [int(median_x + img1.shape[1] / (height_coefficient / mask_correction)), int(median_y + img1.shape[1] / (height_coefficient / mask_correction))]
    # print_map(img1, point, 25)
    # print_map(img1, point2, 25)
    # print_map(img1, point3, 25)
    # print_map(img1, point4, 25)

    return correct_matches, correct_matches_index


def check_matches(main_matches, crop_matches):
    matches_1, matches_2 = [], []
    cnt = 0
    for i in range(len(main_matches)):
        flag = True
        for j in range(len(matches_1)):
            if main_matches[i] == matches_1[j] or crop_matches[i] == matches_2[j]:
                flag = False
                cnt += 1
                break
        if flag:
            matches_1.append(main_matches[i])
            matches_2.append(crop_matches[i])
    # print(f"{cnt=}")
    print(f"Общих точек: до {len(main_matches), len(crop_matches)}, после отсеивания {len(matches_1), len(matches_2)}")
    # print("Undo check")
    # print(len(main_matches), main_matches)
    # print(len(crop_matches), crop_matches, "\n")
    #
    # print("After check")
    # print(len(matches_1), matches_1)
    # print(len(matches_2), matches_2, "\n")
    if len(matches_1) > 3:
        return matches_1, matches_2
    else:
        return [], []


def definition_of_blur(height, altitude):
    diff = int(height / altitude)
    if diff <= 5:
        return 5, 5
    elif diff % 2 == 1:
        return diff-2, diff-2
    else:
        return diff - 1, diff - 1


# WK_00005-1
point_main1 = (48.245954, 46.164273)  # Левый верхний угол
point_main2 = (48.238956, 46.166415)  # Правый верхний угол
point_main3 = (48.238237, 46.160394)  # Нижний верхний угол

path_main = 'C:\\My\\Projects\\images\\main\\WK_00002-1.jpg'
img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)
img1 = cv.GaussianBlur(img1, (5, 5), sigmaX=0, sigmaY=0)
temp_img1 = img1.copy()
kp1, des1 = search_KP(img1)
height_map = 200  # Высота съемки карты
print(f"kp1 {len(kp1)}")

determ = dt.Determ_coord(point_main1, point_main2, point_main3, img1.shape)
filter = None

# Видео с дрона
path_video = 'C:\\My\\Projects\\images\\move5.mp4'
cap = cv.VideoCapture(path_video)

frame_count = 0  # int(cap.get(cv.CAP_PROP_FPS))
f_cnt = 0
coord_cnt = 0
thresh = 0.0015

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        print("Конец видеофайла.")
        break

    if frame_count == 80:
        f_cnt += 1
        frame_count = 0

        # Предобработка кадра
        flight_altitude = 30  # текущая высота полета
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = resize_img(gray, 1024)
        kernel = definition_of_blur(height_map, flight_altitude)
        # kernel = (15, 15)
        print(f"{kernel=}")
        gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)

        # Нахождение опорных точек на кадре и сравнение с опорным изображением
        kp2, des2 = search_KP(gray)
        print(f"\nkp2 {len(kp2)}")
        good_matches = matcher(des1, des2)

        if good_matches != None:
            # поиск общих КТ на главном изображении
            main_matches = location_images(good_matches, kp1)
            main_matches, matches_index = pixel_mask(main_matches)
            matches_2 = location_images_2(good_matches, kp2, matches_index)
            main_matches_filter, matches_2_filter = check_matches(main_matches, matches_2)

            if len(main_matches) > 3:
                coord_cnt += 1
                # center = search_center(main_matches)
                center = true_center(gray, main_matches_filter, matches_2_filter)
                # print_map(img1, center, 30)
                center_coord = determ.calculate(center)
                print_map(img1, center, 30)
                print_map_2(gray, 7)
                img1 = temp_img1.copy()

                # if not filter:
                #     filter = Filter.DroneGPSFilter(center_coord)
                #     # filter = kf.DroneGPSKalman(center_coord)
                #     print_map(center, 20)
                # else:
                #     center_coord_filter = filter.update_coords(center_coord, thresh)
                #     if center_coord_filter != None:
                #         print_map(center, 20)
                #         print(f"Найденное местоположение: Широта = {center_coord[0]}, Долгота = {center_coord[1]}\n")
                #         thresh = 0.0015
                #     else:
                #         thresh += 0.0002
                #         print_map(center, 40)
                #         print(f"Ошибка вычислений: Широта = {center_coord[0]}, Долгота = {center_coord[1]}\n")

                # positions.append(f"{float(center[0])} {float(center[1])}\n")

                # filter.update(center[0], center[1])
                # state_estimate = filter.get_state()
                # filtered.append(state_estimate[:2])
                # print(f"Предсказанное состояние: Широта = {state_estimate}")
                #
                # cv.imshow(" ", gray)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
            else:
                thresh += 0.0002
                print("Совпадений не найдено.\n")
        else:
            thresh += 0.0002

        # # показывает полет наглядно, но тормозит программу
        # plt.imshow(img1, "gray"), plt.show(block=False)
        # plt.pause(0.01)

print(f"Всего кадров: {f_cnt}; найдено: {coord_cnt}")
print(f"{kernel=}")
# filter.visualize(positions, filtered)

# with open("Coordinates.txt", "w") as f:
#     for c in positions:
#         f.write(c)

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
