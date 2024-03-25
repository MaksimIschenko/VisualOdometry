from src.utils import read_image, scale_image, display_image
from src.visual_odometry import find_keypoints_and_descriptors, draw_keypoints, match_keypoints, estimate_motion
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # ДЛЯ XY-plot
    current_position = np.array([0.0, 0.0])
    x_history = [current_position[0]]
    y_history = [current_position[1]]

    # Загрузка параметров камеры
    with open('./src/camera_matrix.json', 'r') as f:
        params = json.load(f)

    # Формирование
    camera_matrix = np.array([
        [params['f_x'], 0, params['c_x']],
        [0, params['f_y'], params['c_y']],
        [0, 0, 1]
    ])

    video_path = './media/office_video_1.mp4'
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео")
        exit()

    # Масштабирование и предварительная обработка первого кадра
    prev_frame = scale_image(prev_frame, 75)
    prev_keypoints, prev_descriptions = find_keypoints_and_descriptors(prev_frame)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # Масштабирование и предварительная обработка текущего кадра
        frame = scale_image(frame, 75)
        keypoints, descriptions = find_keypoints_and_descriptors(frame)

        # Поиск сопоставлений между текущим и предыдущим кадрами
        matches = match_keypoints(prev_descriptions, descriptions)

        # Оценка перемещения между кадрами
        R, t = estimate_motion(matches, prev_keypoints, keypoints, camera_matrix)

        # Аккумуляция смещений и обновление истории координат
        current_position += t[:2].reshape(2,)
        x_history.append(current_position[0])
        y_history.append(current_position[1])

        # Отрисовка ключевых точек на текущем кадре
        frame_with_keypoints = draw_keypoints(frame, keypoints)

        # Отображение обработанного кадра
        cv2.imshow("Processed Frame", frame_with_keypoints)

        # Обновление предыдущих значений
        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptions = descriptions

        # Добавление возможности выхода по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрытие окна вывода видеопотока
    cap.release()
    cv2.destroyAllWindows()

    # Построение траектории движения
    plt.plot(x_history, y_history, marker='o', linestyle='-')
    plt.title('Траектория движения в горизонтальной плоскости')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')  # Установка одинакового масштаба осей для корректного отображения
    plt.show()
