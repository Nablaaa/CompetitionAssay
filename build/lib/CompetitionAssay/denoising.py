"""
Author: Eric Schmidt

This file contains all classes that help to denoise images.
"""


from typing import Any
from n2v.models import N2V
import matplotlib.pyplot as plt
import numpy as np
from CompetitionAssay.datahandling import NormalizeImg


class n2vDenoising:
    """
    Use a pretrained model to denoise images
    First load the model, then load the data, normalize the data
    modify the shape and perform the predictions
    Visualize the results if wanted
    return the predictions
    """

    def __init__(self, model_name: str, model_dir: str) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        self.model = N2V(config=None, name=self.model_name, basedir=self.model_dir)

    def normalize_img(self):
        self.img = NormalizeImg(self.img)

    def modify_shape(self):
        self.img = np.expand_dims(self.img, axis=-1)

    def predict(self, img: np.ndarray):
        self.img = img
        self.normalize_img()
        self.modify_shape()
        self.pred = self.model.predict(self.img, axes="YXC")

        return self.pred[:, :, 0]

    def visualize_img(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(self.img)
        axs[0].set_title("input")

        axs[1].imshow(self.pred[:, :, 0])
        axs[1].set_title("output")

        plt.show()
