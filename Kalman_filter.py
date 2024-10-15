import numpy as np
import matplotlib.pyplot as plt

# class Kalman_filter():
#     # Списки для хранения состояния и измерений для графика
#     predictions = []
#     positions = []
#
#     def __init__(self):
#         # Начальное состояние: [позиция, скорость]
#         self.x = np.array([[0], [0]])
#         # Ковариационная матрица (неопределенность)
#         self.P = np.array([[1000, 0], [0, 1000]])
#         # Модель состояния (переходная матрица)
#         self.F = np.array([[1, 1], [0, 1]])
#         # Модель измерения (матрица измерения)
#         self.H = np.array([[1, 0]])
#         # Ошибка измерения
#         self.R = np.array([[1]])
#         # Единичная матрица
#         self.I = np.eye(2)
#
#     def predict(self):
#         # Прогнозирование
#         self.x = self.F @ self.x
#         self.P = self.F @ self.P @ self.F.T
#
#     def update(self, measurement):
#         # Обновление с новым измерением
#         y = measurement - (self.H @ self.x)  # Разница между измерением и предсказанием
#         S = self.H @ self.P @ self.H.T + self.R  # Ковариация наблюдения
#         K = self.P @ self.H.T @ np.linalg.inv(S)  # Коэффициент Калмана
#
#         # Новое состояние
#         self.x = self.x + K @ y
#         # Обновление ковариационной матрицы
#         self.P = (self.I - K @ self.H) @ self.P
#
#         self.predictions.append([self.x])  # Предсказанная позиция
#         self.positions.append(measurement)  # Фактическая измеренная позиция
#
#
#     def estimate(self, measurement):
#         self.predict()  # Прогнозируем следующее состояние
#         self.update(measurement)  # Обновляем состояние с новыми данными
#         return self.x.flatten()  # Возвращаем состояние в виде одномерного массива
#
#
#     def view_result(self):
#         print(f"{self.positions=}")
#         print(f"{self.predictions=}")
#         # Визуализация результатов
#         plt.plot(self.positions[0], 'ro-', label='Широта найденная')
#         plt.plot(self.positions[1], 'bo-', label='Долгота найденная')
#         plt.legend()
#         plt.xlabel('Долгота')
#         plt.ylabel('Широта')
#         plt.title('Фильтр Калмана')
#         plt.show()

class KalmanFilter:
    def __init__(self, initial_coords, process_variance=1e-5, measurement_variance=1e-2):
        """
        Инициализация фильтра Калмана с начальными координатами.

        initial_coords: начальные координаты в формате [широта, долгота]
        process_variance: дисперсия процесса (шум системы)
        measurement_variance: дисперсия измерений (шум измерений)
        """
        self.state = np.array(initial_coords, dtype=float)  # Начальное состояние [широта, долгота]
        self.uncertainty = np.eye(2) * measurement_variance  # Начальная ковариация
        self.process_variance = process_variance  # Шум процесса
        self.measurement_variance = measurement_variance  # Шум измерений

    def predict(self):
        """
        Предсказание следующего состояния (в данном случае состояние не меняется, так как мы не знаем скорость).
        """
        # В этом примере предсказанное состояние равно предыдущему состоянию
        self.uncertainty += self.process_variance  # Увеличиваем неопределенность (шум процесса)

    def update(self, measurement):
        """
        Обновление состояния фильтра Калмана с новыми измерениями.

        measurement: измеренные координаты в формате [широта, долгота]
        """
        measurement = np.array(measurement, dtype=float)
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
        self.state += kalman_gain @ (measurement - self.state)
        self.uncertainty *= (1 - kalman_gain)

    def get_current_state(self):
        self.state[0] = round(self.state[0], 6)
        self.state[1] = round(self.state[1], 6)
        return self.state.tolist()


class DroneGPSKalman:
    def __init__(self, initial_coords, threshold=0.001):
        """
        Инициализация класса с начальными координатами и фильтром Калмана.

        initial_coords: начальные координаты в формате [широта, долгота]
        threshold: допустимое отклонение для распознавания ошибки (в градусах)
        """
        self.kalman_filter = KalmanFilter(initial_coords)
        self.threshold = threshold

    def update_coords(self, new_coords, threshold):
        self.threshold = threshold
        # Сначала выполняем предсказание нового состояния
        self.kalman_filter.predict()

        # Проверяем отклонение от текущего состояния
        current_state = self.kalman_filter.get_current_state()
        deviation = np.linalg.norm(np.array(new_coords) - np.array(current_state))
        # print(f"{deviation=}")

        if deviation > self.threshold:
            # print(f"Найденное местоположение: Широта = {new_coords[0]}, Долгота = {new_coords[1]}")
            # print(f"Ошибка вычислений: Широта = {current_state[0]}, Долгота = {current_state[1]}")
            return None  # Если отклонение больше порога, возвращаем None

        # Обновляем фильтр новыми измерениями
        self.kalman_filter.update(new_coords)

        # Возвращаем обновленные координаты
        return self.kalman_filter.get_current_state()

        # # Построение графиков результатов
        # plt.figure(figsize=(10, 6))
        # plt.plot(*zip(*true_positions), label='Истинный путь', color='green', marker='x')
        # plt.plot(filtered_lon, filtered_lat, label='Фильтрованная траектория', color='blue', marker='o')
        # plt.scatter(*zip(*measurements), color='red', label='Измерения', s=50, alpha=0.6)
        # plt.title('Использование фильтра Калмана для оценки положения беспилотника')
        # plt.xlabel('Долгота')
        # plt.ylabel('Широта')
        # plt.legend()
        # plt.grid(True)
        # plt.show()