import os
import operator
from PIL import Image
import numpy as np

from src.helpers.data_helper import DataHelper


class PictureHelper:

    @staticmethod
    def open_image(pic_path: str) -> None:
        try:
            with Image.open(pic_path) as image:
                image.show()
        except BaseException as err:
            print(err)
    
    @staticmethod
    def post_array(token: str, fraction: int=3, n_picture: int=2) -> list:
        pictures = []
        directory = DataHelper.picture_path(token)
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    # we have a picture here
                    pictures.append(file_path)
        else:
            raise Exception('What!')
        res = []
        for picture in pictures[:n_picture]:
            res = [*res, *PictureHelper.image_array(picture, fraction)]
        return res
    
    @staticmethod
    def image_array(filepath: str, fraction: int) -> list:
        image = Image.open(filepath)
        n, m = image.size
        data = np.array(list(image.getdata()), dtype=tuple).reshape((n, m, 3))
        nn, mm = map(int, map(operator.mul, (n, m), (1./fraction, 1./fraction)))
        rgbs = np.zeros((3, fraction ** 2))
        for i in range(fraction):
            for j in range(fraction):
                rgb = data[i * nn:(i + 1) * nn, j * mm: (j + 1) * mm, :].reshape(-1, 3)
                rgb = np.mean(rgb, axis=0)
                rgb = rgb.reshape(3, 1)
                rgbs[:, i * fraction + j] = rgb[:, 0]
        rgbs = rgbs.reshape(-1)
        return list(rgbs)

    @staticmethod
    def image_variances(token: str) -> list:
        pictures = []
        directory = DataHelper.picture_path(token)
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # we have a picture here
                pictures.append(file_path)
        res = []
        for picture in pictures:
            res = [*res, tuple(PictureHelper.image_array(picture, 1))]
        if len(res) <= 0:
            return []
        return np.var(np.array(res), axis=0)

