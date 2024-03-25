from enum import Enum


class COLOR(Enum):
    """
    Цвета
    """
    RED = (0, 0, 255)
    GREEN = (0, 117, 16)
    BLUE = (153, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


class VALID_IMG_EXTENSIONS(Enum):
    """
    Верные расширения файлов изображений
    """
    JPG = '.jpg'
    JPEG = '.jpeg'
    PNG = '.png'
    TIFF = '.tiff'
    BMP = '.bmp'
    GIF = '.gif'