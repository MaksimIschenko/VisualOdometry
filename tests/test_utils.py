import numpy as np
import pytest
from src.utils import *

# Пути к тестовым файлам
VALID_IMAGE_PATH = "tests/media/test_office.jpg"
INVALID_EXTENSION_IMAGE_PATH = "tests/media/test_office.txt"


def test_read_image_valid():
    """Тест чтения изображения с допустимым расширением"""
    image = read_image(VALID_IMAGE_PATH)
    assert isinstance(image, np.ndarray), "Функция должна возвращать np.ndarray"


def test_read_image_invalid_extension():
    """Тест на генерацию исключения при недопустимом расширении файла"""
    with pytest.raises(ValueError) as exc_info:
        read_image(INVALID_EXTENSION_IMAGE_PATH)
    assert "Неверное расширение файла" in str(exc_info.value), ("Должно быть сгенерировано исключение с сообщением о "
                                                                "неверном расширении файла")


def test_scale_up_image():
    """Тест увеличения изображения."""
    original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    scale_percent = 200  # Увеличение в 2 раза
    scaled_image = scale_image(original_image, scale_percent)
    assert scaled_image.shape == (200, 200, 3)


def test_scale_down_image():
    """Тест уменьшения изображения."""
    original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    scale_percent = 50  # Уменьшение в 2 раза
    scaled_image = scale_image(original_image, scale_percent)
    assert scaled_image.shape == (50, 50, 3)


def test_scale_image_no_change():
    """Тест без изменения размера изображения."""
    original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    scale_percent = 100  # Без изменений
    scaled_image = scale_image(original_image, scale_percent)
    assert scaled_image.shape == original_image.shape


def test_scale_image_invalid_input():
    """Тест на некорректный ввод."""
    with pytest.raises(AttributeError):
        scale_image(None, 100)  # Передача None вместо изображения


def test_scale_image_invalid_scale_percent():
    """Тест на некорректный масштаб."""
    original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        scale_image(original_image, "50")
