"""
Author: Eric Schmidt

This file contains all classes that help to denoise images.
"""


from n2v.models import N2V
import matplotlib.pyplot as plt
import numpy as np
from CompetitionAssay.datahandling import NormalizeImg


def n2vDenoising(img, visualize=False):
    # normalize image
    img = NormalizeImg(img)

    # We import all our dependencies.
    model_name = "n2v_transwell"
    model_dir = "models"
    model = N2V(config=None, name=model_name, basedir=model_dir)

    # change shape to (y,x, c=1)
    img = np.expand_dims(img, axis=-1)

    pred = model.predict(img, axes="YXC")

    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        axs[0].imshow(img)
        axs[0].set_title("input")

        axs[1].imshow(pred[:, :, 0])
        axs[1].set_title("output")

        plt.show()

    return pred[:, :, 0]  # return same shape as input

