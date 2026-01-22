import cv2 as cv
import time
import Comparator
import Preprocessing
import SearchMethods
import DetermCoord


class Testing:
    """
    Основной класс для тестирования алгоритмов определения местоположения
    по опорному изображению и видеопотоку или отдельному кадру.
    """
    path_main = r"C:\My\Projects\images\main\WK_00005-1.jpg"
    img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)

    gray = cv.imread(r"C:\My\Projects\images\main\WK_00002-1.jpg", cv.IMREAD_GRAYSCALE)
    gray = Preprocessing.resize_img(gray, 1024)

    path_video = r"C:\My\Projects\images\move3.mp4"
    visual = 0          # Визуализация найденных совпадений

    flight_altitude = 30    # текущая высота полета
    height_map = 500        # Высота съемки карты
    frame_count = 0         # int(cap.get(cv.CAP_PROP_FPS))
    f_cnt = 0
    coord_cnt = 0
    method_index = 1    # 1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"

    # Координаты углов опорного изображения для WK_00005-1.jpg
    point_main1 = (48.245954, 46.164273)  # Левый верхний угол
    point_main2 = (48.238956, 46.166415)  # Правый верхний угол
    point_main3 = (48.238237, 46.160394)  # Нижний правый угол

    def __init__(self, format_video=True):
        """
        Инициализация тестового запуска.
        :param format_video: True — обработка видео; False — обработка одного изображения.
        """
        cv.imwrite("main_with_points_new.jpg", self.img1)
            
        # Создание карт по изображениям и угловым координатам
        self.determ_main = DetermCoord.DetermCoord(self.point_main1, self.point_main2,
                                                   self.point_main3, self.img1.shape)

        self.img1 = cv.GaussianBlur(self.img1, (5, 5), sigmaX=0, sigmaY=0)

        self.method = SearchMethods.Method(self.method_index, 0.48)
        if format_video:
            self.main_video_cycle()
        else:
            self.main_with_image()


    def selection_of_blur(self, height, altitude):
        """
        Автоматический выбор размера ядра фильтра Гаусса в зависимости от отношения
        высоты съёмки карты к текущей высоте полёта.
        Цель — имитировать размытие, соответствующее масштабу изображения.
        :param height: высота съёмки опорной карты
        :param altitude: текущая высота полёта
        :return: кортеж (ksize_x, ksize_y) — размер ядра (нечётные числа)
        """
        diff = int(height / altitude)
        if diff <= 7:
            return 7, 7
        elif diff % 2 == 1:
            return diff - 2, diff - 2
        else:
            return diff - 1, diff - 1


    def main_video_cycle(self):
        """
        Основной цикл обработки видео:
        - читает видео по кадрам,
        - каждые 30 кадров (примерно 1 сек при 30 FPS) обрабатывает один кадр,
        - сравнивает его с опорным изображением,
        - вычисляет геокоординаты и записывает в файл.
        """
        cap = cv.VideoCapture(self.path_video)
        kp1, des1 = self.method.get_kp_and_des(self.img1)
        start = time.perf_counter()

        while cap.isOpened():
            self.frame_count += 1
            ret, frame = cap.read()
            
            if not ret:
                print("Конец видеофайла.")
                break

            if self.frame_count == 30:
                self.f_cnt += 1
                self.frame_count = 0

                # Предобработка кадра
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray = Preprocessing.resize_img(gray, 1024)
                kernel = self.selection_of_blur(self.height_map, self.flight_altitude)
                # kernel = (5, 5)
                gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)

                test = Comparator.Compare(
                    main_img=self.img1,
                    kp_main=kp1,
                    des_main=des1,
                    height_main=self.height_map,
                    img_2=gray,
                    altitude=self.flight_altitude,
                    method=self.method
                )

                location = test.comparator(self.visual)
                if location:
                    self.coord_cnt += 1
                    coord = self.determ_main.calculate(location)
                    print(coord)

        finish = time.perf_counter()
        minutes = round((finish - start) // 60)
        seconds = round((finish - start) % 60, 2)
        print(f"\nВремя выполнения программы: {minutes} мин. {seconds} сек.")
        print(f"Всего кадров: {self.f_cnt}; найдено: {self.coord_cnt}")


    def main_with_image(self):
        """
        Режим обработки одного изображения (для отладки или тестирования).
        Аналогичен видеоциклу, но без цикла и с визуализацией результатов.
        """
        kp1, des1 = self.method.get_kp_and_des(self.img1)

        # kernel = self.selection_of_blur(self.height_map, self.flight_altitude)
        # # kernel = (5, 5)
        # gray = cv.GaussianBlur(self.gray, kernel, sigmaX=0, sigmaY=0)

        test = Comparator.Compare(
            main_img=self.img1,
            kp_main=kp1,
            des_main=des1,
            height_main=self.height_map,
            img_2=self.gray,
            altitude=self.flight_altitude,
            method=self.method
        )

        location = test.comparator(self.visual)
        coord = self.determ_main.calculate(location) if location else "Положение не обнаружено"
        print(coord)


if __name__ == "__main__":
    test = Testing(format_video=1)