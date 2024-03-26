import sys
import cv2
import numpy as np
import json
import glob
from typing import Tuple, List
from src.utils import display_image, read_image
from src.constants import COLOR


def find_corners(image: np.ndarray, checkerboard_size: (Tuple[int, int]),
                 square_size: float) -> (np.ndarray | None, np.ndarray | None):
    """
    Определяет и возвращает угловые точки шахматной доски на изображениях для калибровки камеры.


    :param checkerboard_size: размер шахматной доски (кол-во квадратов по вертикали и горизонтали)
    :param square_size: размер квадрата шахматной доски
    :param image: изображение представленное в numpy массиве

    :return: (objpoints, imgpoints): Первый список (`objpoints`) содержит массивы трехмерных точек для каждой
    шахматной доски, успешно обнаруженной на изображениях. Второй список (`imgpoints`) содержит массивы двумерных
    точек, соответствующих углам шахматных досок на изображениях.
    """

    # Подготовка точек в трехмерном пространстве
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    # Находим углы на шахматной доске
    ret, corners = cv2.findChessboardCorners(image, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)

    if ret:
        return objp, corners

    return None, None


def draw_corners(image: np.ndarray, objp: np.ndarray | None, imgp: np.ndarray | None) -> np.ndarray:
    """
    Отрисовка точек на исх. изображении, отображающая определённые углы
    :param image: изображение, представленное в numpy массиве
    :param objp: массив трехмерных точек
    :param imgp: массивы двумерных точек
    :return: изображение, представленное в numpy массиве с отрисованными углами
    """
    if objp is not None:
        for point in imgp:
            x, y = point[0].astype(int)
            cv2.circle(image, (x, y), 10, COLOR.WHITE.value, -1)
    else:
        cv2.putText(image, "Not found", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, COLOR.WHITE.value, 2)
    return image


def get_camera_param(image: np.ndarray, objp: List[np.ndarray], imgp: List[np.ndarray]):
    """
    Вычисляет параметры камеры, используя калибровку по методу наименьших квадратов
    на основе трехмерных точек в реальном мире и соответствующих двумерных точек на изображениях.

    :param image: изображение (для определения размера)
    :param objp: списки трехмерных точек
    :param imgp: список двумерных точек
    :return: Словарь, содержащий внутренние параметры камеры. Включает в себя:
            - f_x (float): Фокусное расстояние камеры по оси X.
            - c_x (float): Координата центра проекции по оси X.
            - f_y (float): Фокусное расстояние камеры по оси Y.
            - c_y (float): Координата центра проекции по оси Y.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objp,
                                                                        imagePoints=imgp,
                                                                        imageSize=image.shape[::-1],
                                                                        cameraMatrix=None,
                                                                        distCoeffs=None)
    params = {}
    params['f_x'] = camera_matrix[0][0]
    params['c_x'] = camera_matrix[0][2]
    params['f_y'] = camera_matrix[1][1]
    params['c_y'] = camera_matrix[1][2]

    return params


def write_to_json(dictionary, filename):
    """
    Запись словаря в json файл
    :param dictionary: словарь (type dict)
    :param filename: название файла
    """
    with open(filename, 'w') as f:
        json.dump(dictionary, f)


if __name__ == "__main__":

    image_path = '../media/calibration/*.jpg'
    checkerboard_size = (7, 7)
    square_size = 33.0

    images = glob.glob(image_path)

    img_to_calib = []
    objpoints = []
    imgpoints = []

    for i in range(len(images)):

        # Чтение изображения
        img = read_image(images[i], mode=0)

        # Поиск углов изображения
        objp, imgp = find_corners(image=img,
                                  checkerboard_size=checkerboard_size,
                                  square_size=square_size)

        # Если углы не определены, то переход к следующей интерации
        if objp is None or imgp is None:
            continue

        # Формирование списков для функция вычисления параметров
        img_to_calib.append(img)
        objpoints.append(objp)
        imgpoints.append(imgp)

        # Опционально. Отрисовка углов на исх. изображении
        img = draw_corners(img, objp, imgp)
        display_image(img)

    # Получение параметров камеры
    params = get_camera_param(imgp=imgpoints,
                              objp=objpoints,
                              image=img_to_calib[-1])

    # Запись параметров в json файл
    write_to_json(params, '../src/camera_matrix.json')
