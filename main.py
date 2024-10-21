import cv2 as cv
import time
import Comparator
import Comparator_new
import Preprocessing
import SearchMethods


class Testing_video():
    path_main = 'C:\\My\\Projects\\images\\main\\WK_00005-1.jpg'
    img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)
    # Видео с дрона
    path_video = 'C:\\My\\Projects\\images\\move5.mp4'
    cap = cv.VideoCapture(path_video)

    flight_altitude = 30  # текущая высота полета
    height_map = 500  # Высота съемки карты

    frame_count = 0  # int(cap.get(cv.CAP_PROP_FPS))
    f_cnt = 0
    coord_cnt = 0
    version = 2  # 1-old; 2-new

    def __init__(self):
        # Обновление изображения для отображения всех точек
        if self.version == 1:
            cv.imwrite("main_with_points.jpg", self.img1)
        else:
            cv.imwrite("main_with_points_new.jpg", self.img1)
        self.img1 = cv.GaussianBlur(self.img1, (5, 5), sigmaX=0, sigmaY=0)
        # 1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"
        self.method = SearchMethods.Method(1, 0.48)

    def definition_of_blur(self, height, altitude):
        diff = int(height / altitude)
        if diff <= 5:
            return 5, 5
        elif diff % 2 == 1:
            return diff - 2, diff - 2
        else:
            return diff - 1, diff - 1

    def main_cycle(self):
        start = time.perf_counter()
        kp1, des1 = self.method.get_kp_and_des(self.img1)

        # gray = cv.imread("C:\\My\\Projects\\images\\main\\WK_00002-1.jpg", cv.IMREAD_GRAYSCALE)
        # gray = Preprocessing.resize_img(gray, 1024)
        # kernel = self.definition_of_blur(self.height_map, 200)
        # gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)
        #
        # test = Comparator.Compare(self.img1, kp1, des1, self.height_map, gray, 200, self.method)
        # self.coord_cnt += test.comparator(0)

        while self.cap.isOpened():
            self.frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                print("Конец видеофайла.")
                break

            if self.frame_count == 30:
                self.f_cnt += 1
                print("№ кадра -", self.f_cnt)
                self.frame_count = 0

                # Предобработка кадра
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray = Preprocessing.resize_img(gray, 1024)
                kernel = self.definition_of_blur(self.height_map, self.flight_altitude)
                gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)

                if self.version == 1:
                    test = Comparator.Compare(self.img1, kp1, des1, self.height_map, gray, self.flight_altitude,
                                              self.method)
                else:
                    test = Comparator_new.Compare(self.img1, kp1, des1, self.height_map, gray, self.flight_altitude,
                                               self.method)

                self.coord_cnt += test.comparator(visual=0)

        finish = time.perf_counter()
        minutes = round((finish - start) // 60)
        seconds = round((finish - start) % 60)
        print(f"\n{kernel=}")
        print(f"\nВремя выполнения программы: {minutes} мин. {seconds} сек.")
        print(f"Всего кадров: {self.f_cnt}; найдено: {self.coord_cnt}")


if __name__ == "__main__":
    test = Testing_video()
    test.main_cycle()
