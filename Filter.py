import numpy as np


class DroneGPSFilter:
    def __init__(self, initial_coords, threshold=0.001):
        self.previous_coords = np.array(initial_coords)
        self.threshold = threshold

    def is_outlier(self, new_coords):
        new_coords = np.array(new_coords)
        # Вычисление расстояния между предыдущими и новыми координатами
        distance = np.linalg.norm(new_coords - self.previous_coords)
        # Если расстояние больше порогового значения, считаем это ошибкой
        return distance > self.threshold

    def update_coords(self, new_coords, threshold):
        self.threshold = threshold

        if self.is_outlier(new_coords):
            return None
        else:
            self.previous_coords = np.array(new_coords)
            return new_coords


