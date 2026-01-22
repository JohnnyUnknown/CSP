import cv2 as cv
import Preprocessing
import numpy as np


class Compare:
    """
    Класс для сравнения опорного изображения (карты) с текущим кадром (с БПЛА)
    и определения географического положения через ключевые точки и гомографию.
    """
    def __init__(self, *, main_img, kp_main, des_main, height_main, img_2, altitude, method):
        self.good_match = 0
        self.filter_matches = 0
        self.center = None
        self.center_location = None
        self.key_1 = 0       # Кол-во контрольных точек основного изображения
        self.key_2 = 0       # Кол-во контрольных точек области видимости

        self.img1 = main_img  # print_map
        self.kp1 = kp_main
        self.des1 = des_main
        self.height_map = height_main
        self.img_size = main_img.shape
        self.gray = img_2
        self.flight_altitude = altitude
        self.method = method    # Объект класса Method


    def main_frame_points(self, good_matches):
        """
        Извлекает координаты совпадающих ключевых точек на опорном изображении.
        """
        main_matches = []
        for i in range(len(good_matches)):
            dmatch = good_matches[i]
            # Поиск найденных КТ для обеих изображений в списке КТ главного изображения
            large_image_KP = list(self.kp1[dmatch.queryIdx].pt)
            large_image_KP[0] = int(large_image_KP[0])
            large_image_KP[1] = int(large_image_KP[1])
            # Добавление в список КТ главного изображения, совпадающих с КТ искомого
            main_matches.append(large_image_KP)
        return main_matches


    def frame_points(self, good_matches, kp, main_matches_index):
        """
        Извлекает координаты совпадающих КТ на текущем кадре, но ТОЛЬКО для индексов из matches_index.
        """
        frame_matches = []
        for i in range(len(good_matches)):
            if i in main_matches_index:
                large_image_KP = list(kp[good_matches[i].trainIdx].pt)
                large_image_KP[0] = int(large_image_KP[0])
                large_image_KP[1] = int(large_image_KP[1])
                frame_matches.append(large_image_KP)
        return frame_matches


    def print_map(self):
        """
        Визуализация результата: центр на карте и на кадре.
        """
        # Отрисовка найденного центра на опороном изображении
        color = (0, 255, 0)
        temp_main_img = self.img1.copy()
        main_img = cv.circle(temp_main_img, self.center, radius=4, color=color, thickness=25)
        center2 = [round(self.gray.shape[1] / 2), round(self.gray.shape[0] / 2)]
        crop_img = cv.circle(self.gray, center2, radius=3, color=color, thickness=6)

        cv.imshow("Main image", Preprocessing.resize_img(main_img, 1024))
        if crop_img.shape[1] > 1024:
            new_width = 1024
        else:
            new_width = crop_img.shape[1]
        cv.imshow("Crop image", Preprocessing.resize_img(crop_img, new_width))
        cv.waitKey(0)
        cv.destroyAllWindows()


    def pixel_mask(self, matches):
        """
        Фильтрует совпадающие ключевые точки на опорном изображении,
        оставляя только те, что находятся в пределах ожидаемой области видимости
        текущего кадра, спроецированного на карту.

        Возвращает:
            - correct_matches: список отфильтрованных точек.
            - correct_matches_index: их исходные индексы в списке `matches`.
        """
        if not matches:
            return [], []

        matches_arr = np.array(matches)  
        # Медианное положение совпадающих точек 
        median_x = np.median(matches_arr[:, 0])
        median_y = np.median(matches_arr[:, 1])

        # Коэффициент масштабирования: во сколько раз карта "крупнее" текущего кадра
        scale_factor = self.height_map / self.flight_altitude
        max_deviation = max(self.img1.shape[:2]) / scale_factor

        mask = (
            (matches_arr[:, 0] >= median_x - max_deviation) &
            (matches_arr[:, 0] <= median_x + max_deviation) &
            (matches_arr[:, 1] >= median_y - max_deviation) &
            (matches_arr[:, 1] <= median_y + max_deviation)
        )

        # Применяем маску
        correct_matches = matches_arr[mask]
        correct_matches_index = np.where(mask)[0]

        print(f"Before mask: {len(matches)} -> After mask: {len(correct_matches)}")
        return correct_matches, correct_matches_index


    def transformation_matrix(self, main_matches, matches_2):
        """
        Вычисляет гомографию между кадром и картой.
        """
        if len(main_matches) < 4 or len(matches_2) < 4:
            return None
        # Массивы с точками соответствия
        pts1 = np.float32([m for m in matches_2]).reshape(-1, 1, 2)
        pts2 = np.float32([m for m in main_matches]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
        return H


    def true_center(self, img, main_matches, frame_matches):
        """
        Преобразует центр кадра в координаты на карте через гомографию.
        """
        crop_center = np.array([[img.shape[1] / 2, img.shape[0] / 2]], dtype='float32').reshape(-1, 1, 2)
        H = self.transformation_matrix(main_matches, frame_matches)
        try:
            find_center = cv.perspectiveTransform(crop_center, H)
            true_center = []
            true_center.append(int(find_center[0][0][0]))
            true_center.append(int(find_center[0][0][1]))

            return true_center

        except cv.error:
            print("Ошибка матрицы гомографии.\n")
            return None


    def comparator(self, visual=True):
        """
        Основной метод: выполняет сопоставление, фильтрацию, гомографию и возвращает центр.
        """
        # Если выбран метод 5 (SuperPoints), вернется (None, None)
        kp2, des2 = self.method.get_kp_and_des(self.gray)
        
        if kp2 == None:
            self.kp1, kp2, good_matches = self.method.find_and_get_matches(img1=self.img1, img2=self.gray)
        else:
            _, _, good_matches = self.method.find_and_get_matches(des1=self.des1, des2=des2)

        print(f"\n1: {len(self.kp1)}, 2: {len(kp2)}, Общих: {None if good_matches == None else len(good_matches)}")
        if good_matches != None:
            main_matches = self.main_frame_points(good_matches)
            # matches_index = [i for i in range(len(main_matches))]
            main_matches, matches_index = self.pixel_mask(main_matches)
            frame_matches = self.frame_points(good_matches, kp2, matches_index)

            if len(main_matches) > 3:
                self.center = self.true_center(self.gray, main_matches, frame_matches)
                if self.center:
                    if visual:
                        self.print_map()

                    # Отрисовка найденного центра и сохрание изображения в файл
                    main_img_with_points = cv.circle(cv.imread("main_with_points_new.jpg"),
                                                        self.center,
                                                        radius=10,
                                                        color=(0, 0, 0),
                                                        thickness=20
                                                        )
                    cv.imwrite("main_with_points_new.jpg", main_img_with_points)

                    return self.center
        return None
