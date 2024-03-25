import cv2
import numpy as np
import os
from src.constants import VALID_IMG_EXTENSIONS


def read_image(image_path: str, mode: int = 1) -> np.ndarray:
    """
    Чтение изображения

    :param image_path: путь до изображения
    :param mode: 0 - без преобразования в GREY / 1 - c преобразованием в GREY

    :return: изображение представленное массивом numpy
    """

    # Проверка расширения файла
    file_ext = os.path.splitext(image_path)[1]
    if file_ext not in [ext.value for ext in VALID_IMG_EXTENSIONS]:
        raise ValueError(f"Неверное расширение файла: {file_ext}.")
    if mode == 0:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def scale_image(image: np.ndarray, scale_percent: int = 100) -> np.ndarray:
    """
    Масштабирование изображения
    :param image: исходное изображение
    :param scale_percent: масштаб изображения
    :return: масштабированное изображение
    """
    if image is None:
        raise AttributeError("Предоставленное изображение не может быть None.")

    if scale_percent == 100:
        return image

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def display_image(image: np.ndarray) -> None:
    """
    Отображение изображения
    :param image: изображение представленное в numpy массиве
    """
    cv2.imshow('Image', image)
    while True:
        key = cv2.waitKey(0)
        if key == 32:
            break
    cv2.destroyAllWindows()
