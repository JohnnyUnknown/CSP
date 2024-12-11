import cv2 as cv
import time
import Comparator
import Comparator_new
import Preprocessing
import SearchMethods


class Testing_video():
    path_main = 'C:/My/Projects/images/main/MyMap-2.jpg'    # yand_maps_2-2.bmp MyMap-2.jpg WK_00005-1.jpg
    img1 = cv.imread(path_main, cv.IMREAD_GRAYSCALE)
    # img1 = Preprocessing.resize_img(img1, 1024)

    # img1 = cv.Sobel(img1, ddepth=-1, dx=1, dy=1, ksize=5)
    # img1 = cv.Laplacian(img1, ddepth=-1, ksize=3)
    # im = cv.normalize(img1, None, 0, 255, cv.NORM_MINMAX)
    # im = Preprocessing.resize_img(im, 1920)
    # cv.imshow("main image", im)
    # cv.waitKey(0)

    # Видео с дрона
    path_video = "C:/My/Projects/images/move4.mp4"

    flight_altitude = 500  # текущая высота полета
    height_map = 500  # Высота съемки карты

    frame_count = 0  # int(cap.get(cv.CAP_PROP_FPS))
    f_cnt = 0
    coord_cnt = 0
    method_index = 5    # 1: "SIFT", 2: "AKAZE", 3: "ORB", 4: "ASIFT", 5: "SuperPoint"
    version = 2  # 1-old; 2-new

    def __init__(self):
        # Обновление изображения для отображения всех точек
        if self.version == 1:
            cv.imwrite("main_with_points.jpg", self.img1)
        else:
            cv.imwrite("main_with_points_new.jpg", self.img1)
        self.img1 = cv.GaussianBlur(self.img1, (5, 5), sigmaX=0, sigmaY=0)
        self.method = SearchMethods.Method(self.method_index, 0.48)

    def definition_of_blur(self, height, altitude):
        diff = int(height / altitude)
        if diff <= 5:
            return 5, 5
        elif diff % 2 == 1:
            return diff - 2, diff - 2
        else:
            return diff - 1, diff - 1

    def main_video_cycle(self):
        cap = cv.VideoCapture(self.path_video)
        start = time.perf_counter()
        kp1, des1 = self.method.get_kp_and_des(self.img1)

        while cap.isOpened():
            self.frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print("Конец видеофайла.")
                break

            if self.frame_count == 30:
                self.f_cnt += 1
                # print("№ кадра -", self.f_cnt)
                self.frame_count = 0

                # Предобработка кадра
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray = Preprocessing.resize_img(gray, 1024)
                kernel = self.definition_of_blur(self.height_map, self.flight_altitude)
                # kernel = (5, 5)
                gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)
                # gray = cv.Laplacian(gray, ddepth=-1, ksize=3)
                # gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
                # cv.imshow("", gray)
                # cv.waitKey(0)
                # cv.destroyAllWindows()


                if self.version == 1:
                    test = Comparator.Compare(
                        main_img=self.img1,
                        kp_main=kp1,
                        des_main=des1,
                        height_main=self.height_map,
                        img_2=gray,
                        altitude=self.flight_altitude,
                        method=self.method
                    )
                else:
                    test = Comparator_new.Compare(
                        main_img=self.img1,
                        kp_main=kp1,
                        des_main=des1,
                        height_main=self.height_map,
                        img_2=gray,
                        altitude=self.flight_altitude,
                        method=self.method
                    )

                self.coord_cnt += test.comparator(visual=1)

        finish = time.perf_counter()
        minutes = round((finish - start) // 60)
        seconds = round((finish - start) % 60)
        print(f"\n{kernel=}")
        print(f"\nВремя выполнения программы: {minutes} мин. {seconds} сек.")
        print(f"Всего кадров: {self.f_cnt}; найдено: {self.coord_cnt}")

    def main_with_image(self):
        start = time.perf_counter()
        kp1, des1 = self.method.get_kp_and_des(self.img1)

        gray = cv.imread("C:\\My\\Projects\\images\\main\\yand_maps_2.bmp", cv.IMREAD_GRAYSCALE)
        # gray = Preprocessing.resize_img(gray, 1024)
        # kernel = self.definition_of_blur(self.height_map, 500)
        kernel = (5, 5)
        gray = cv.GaussianBlur(gray, kernel, sigmaX=0, sigmaY=0)

        if self.version == 1:
            test = Comparator.Compare(
                main_img=self.img1,
                kp_main=kp1,
                des_main=des1,
                height_main=self.height_map,
                img_2=gray,
                altitude=self.flight_altitude,
                method=self.method
            )
        else:
            test = Comparator_new.Compare(
                main_img=self.img1,
                kp_main=kp1,
                des_main=des1,
                height_main=self.height_map,
                img_2=gray,
                altitude=self.flight_altitude,
                method=self.method
            )
        self.coord_cnt += test.comparator(1)

        finish = time.perf_counter()
        minutes = round((finish - start) // 60)
        seconds = round((finish - start) % 60)
        print(f"\n{kernel=}")
        print(f"\nВремя выполнения программы: {minutes} мин. {seconds} сек.")
        print(f"Всего кадров: {self.f_cnt}; найдено: {self.coord_cnt}")

if __name__ == "__main__":
    test = Testing_video()
    # test.main_video_cycle()
    test.main_with_image()
