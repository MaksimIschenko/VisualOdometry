import numpy as np
import cv2
from matplotlib import pyplot as plt
from src.utils import *

# Загрузите изображение
image = cv2.imread('../media/calibration/2.jpg')
image = scale_image(image, 50)
display_image(image)

# Преобразуйте изображение из BGR в LAB
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Разделите каналы
l, a, b = cv2.split(image_lab)

# Создайте объект CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Примените CLAHE к каналу L
l_clahe = clahe.apply(l)

# Объедините каналы обратно и преобразуйте в BGR
image_lab_clahe = cv2.merge((l_clahe, a, b))
image_color_clahe = cv2.cvtColor(image_lab_clahe, cv2.COLOR_LAB2BGR)

display_image(image_color_clahe)

image = image_color_clahe

# Инициализируйте начальную маску
mask = np.zeros(image.shape[:2], np.uint8)

# Инициализируйте прямоугольник, содержащий передний план (формат: x,y,w,h)
# Этот прямоугольник должен включать объект, который вы хотите выделить
rect = (1, 1, image.shape[1]-1, image.shape[0]-1)  # Пример значения, настройте под ваше изображение

# Инициализация временных массивов, используемых алгоритмом GrabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Примените GrabCut
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
display_image(image)

# Пометьте передний план и возможный передний план как 1, остальное как 0
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

# Создайте изображение только с передним планом
image_fg = image * mask2[:, :, np.newaxis]

display_image(image_fg)