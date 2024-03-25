import pytest
import numpy
from src.visual_odometry import *
from src.utils import read_image

# Пути к тестовым файлам
VALID_IMAGE_PATH = "tests/media/test_office.jpg"
EMPTY_IMG = "tests/media/mono.png"


def test_find_keypoints_and_descriptors_with_valid_image():
    # Читаем тестовое изображение
    test_image = read_image(VALID_IMAGE_PATH)

    keypoints, descriptors = find_keypoints_and_descriptors(test_image)

    # Проверяем, что функция возвращает кортеж из двух элементов
    assert isinstance(keypoints, tuple), "Ключевые точки должны быть кортежем"
    assert isinstance(descriptors, numpy.ndarray), "Описания должны быть массивом numpy"

    # Проверяем, что были найдены ключевые точки
    assert len(keypoints) > 0, "Должна быть найдена хотя бы одна ключевая точка"
    assert descriptors.shape[0] == len(
        keypoints), "Количество описаний должно соответствовать количеству ключевых точек"


def test_find_keypoints_and_descriptors_with_empty_image():
    # Создаем пустое черно-белое изображение
    empty_image = read_image('tests/media/mono.png')

    keypoints, descriptors = find_keypoints_and_descriptors(empty_image)

    # Проверяем, что ключевые точки не были найдены в пустом изображении
    assert len(keypoints) == 0, "В пустом изображении не должно быть ключевых точек"
    # Проверяем, что описания отсутствуют
    assert descriptors is None, "В пустом изображении не должно быть описаний"
