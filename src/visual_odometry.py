import cv2
import numpy as np
from cv2 import DMatch

from src.constants import COLOR
from typing import Optional, Tuple, Any, List


def find_keypoints_and_descriptors(image: np.ndarray) -> (Optional[Tuple[Any, ...]], np.ndarray):
    """
    Поиск ключевых точек и их описание
    :param image:
    :return: кортеж ключевых точек (тип cv2.KeyPoint) и описание изображения
    """
    orb = cv2.ORB_create()  # Oriented FAST and Rotated BRIEF метод
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def draw_keypoints(image: np.ndarray, keypoints: Optional[Tuple[Any, ...]] = ()) -> np.ndarray:
    """
    Визуализации ключевых точек на исходном изображении
    :param image: исходное изображение
    :param keypoints: кортеж ключевых точек
    :return: изображение с ключевыми точками представленное массивом numpy
    """
    if not keypoints:
        return image

    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=COLOR.RED.value, flags=0)
    return image_with_keypoints


def match_keypoints(descriptors1: np.ndarray, descriptors2: np.ndarray) -> Optional[list[DMatch]]:
    """
    Сопоставление ключевых точек по описанию
    :param descriptors1: описание ключевых точек изображения i
    :param descriptors2: описание ключевых точек изображения i+1
    :return: Список сопоставлений
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def estimate_motion(matches: List[cv2.DMatch],
                    keypoints1: List[cv2.KeyPoint],
                    keypoints2: List[cv2.KeyPoint],
                    camera_matrix: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Оценка перемещения между двумя кадрами.
    :param matches: Список сопоставлений между двумя наборами ключевых точек.
    :param keypoints1: Ключевые точки на первом изображении.
    :param keypoints2: Ключевые точки на втором изображении.
    :param camera_matrix: Матрица камеры.
    :return: Кортеж, содержащий векторы вращения и смещения, если перемещение успешно оценено.
    """

    if len(matches) < 4:
        return None

    # Извлечение положений ключевых точек из сопоставлений
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Вычисление матрицы эссенциальных параметров
    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Восстановление позиции и ориентации
    if E is None:
        return None

    # Восстановление позиции и ориентации
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix, mask=mask)
    # векторы вращения и смещения

    return R, t

