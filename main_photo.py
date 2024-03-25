from src.utils import read_image, scale_image, display_image
from src.visual_odometry import find_keypoints_and_descriptors, draw_keypoints, match_keypoints, estimate_motion
import json
import numpy as np
import cv2


if __name__ == "__main__":

    imgs_path = ['media/office_0.jpg', 'media/office_1.jpg']
    imgs = []
    for path in imgs_path:
        pre_scaled = read_image(path)
        imgs.append(scale_image(pre_scaled, 75))

    keypoints_arr = []
    descriptions_arr = []
    for img in imgs:
        keypoints, descriptions = find_keypoints_and_descriptors(img)
        keypoints_arr.append(keypoints)
        descriptions_arr.append(descriptions)

        img = draw_keypoints(img, keypoints)
        display_image(img)

    matches = match_keypoints(descriptions_arr[0], descriptions_arr[1])

    matched_image = cv2.drawMatches(imgs[0], keypoints_arr[0], imgs[1], keypoints_arr[1], matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_image(matched_image)

    with open('./src/camera_matrix.json', 'r') as f:
        params = json.load(f)

    # Формирование массива на основе значений из словаря
    camera_matrix = np.array([
        [params['f_x'], 0, params['c_x']],
        [0, params['f_y'], params['c_y']],
        [0, 0, 1]
    ])

    R, t = estimate_motion(matches, keypoints_arr[0], keypoints_arr[1], camera_matrix)

    print(R)

    print(t)
