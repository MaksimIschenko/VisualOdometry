# Калибровочная матрица камеры

## Определение

Аргумент camera_matrix в функции estimate_motion, которая использует cv2.solvePnPRansac для оценки движения между двумя наборами ключевых точек, представляет собой внутреннюю калибровочную матрицу камеры. 
Эта матрица несет информацию о внутренних параметрах камеры, которые необходимы для точной работы алгоритмов компьютерного зрения, связанных с пространственным позиционированием и 3D-реконструкцией.

Внутренняя калибровочная матрица камеры обычно имеет следующий вид:
```
[f_x,   0,  c_x]
[  0, f_y,  c_y]
[  0,   0,   1 ]
```

где:

- `f_x` и `f_y` — фокусные расстояния камеры в пикселях по осям x и y соответственно. Фокусные расстояния определяют масштаб изображения на сенсоре камеры.
- `c_x` и `c_y` — координаты оптического центра (или главной точки) камеры в пикселях. Оптический центр — это точка на сенсоре камеры, через которую проходит оптическая ось (линия перпендикулярная плоскости сенсора и проходящая через центр линзы).


Эта матрица используется для преобразования координат точек в трехмерном пространстве в координаты на изображении (и наоборот) с учетом особенностей конкретной камеры.

При использовании функции `cv2.solvePnPRansac`, `camera_matrix` помогает корректно связать пространственное положение объектов и их проекции на изображении, что критически важно для точной оценки вращения (`rvec`) и перемещения (`tvec`) между двумя наборами точек, полученными на основе сопоставления ключевых точек двух изображений.

---

## Процесс калибровки

 Калибровка камеры позволяет определить эти параметры, используя серию изображений известных калибровочных шаблонов (чаще всего используется шахматная доска). 
 Вот общие шаги для калибровки камеры и определения этих параметров:
 
1. Соберите калибровочные изображения
Снимите серию изображений калибровочного объекта (например, шахматной доски) под разными углами и на разных расстояниях. 
Убедитесь, что шахматная доска занимает большую часть кадра и что углы хорошо видны.

2. Используйте OpenCV для калибровки камеры
Используйте функции OpenCV, такие как `cv2.findChessboardCorners` и `cv2.calibrateCamera`, для определения внутренних параметров камеры. 
Вам понадобится указать реальные размеры шахматной доски (количество углов и размер квадрата).
```
import cv2
import numpy as np
import glob

# Параметры шахматной доски
checkerboard_size = (6, 9)  # количество углов по горизонтали и вертикали
square_size = 24.0  # размер квадрата в мм или других единицах измерения

# Создание векторов для хранения точек в трехмерном и двумерном пространствах
objpoints = []  # точки в реальном мире
imgpoints = []  # точки на изображении

# Подготовка точек в трехмерном пространстве
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Соберите список калибровочных изображений
images = glob.glob('path/to/calibration/images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Находим углы на шахматной доске
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Калибровка камеры
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```
После выполнения этого кода `camera_matrix` содержит калибровочную матрицу камеры с фокусными расстояниями и координатами оптического центра.

3. Проверка и использование результатов калибровки
После калибровки вы можете использовать полученную матрицу камеры и коэффициенты искажения (`dist_coeffs` в примере кода) для коррекции искажений изображений и для других задач компьютерного зрения, требующих знания внутренних параметров камеры.