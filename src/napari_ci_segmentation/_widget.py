"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
import numpy as np
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from napari.qt.threading import thread_worker

from skimage import (
    color,
    feature,
    filters,
    measure,
    morphology,
    segmentation,
    util,
)

if TYPE_CHECKING:
    import napari


@thread_worker
def segment(image: "napari.types.Image"):
    image_data = np.asarray(image.data)
    image_data = blur_and_gray_image(image_data)
    yield (f"{image.name}_blurred", image_data, True)
    labels = initial_segment(image_data)
    yield (f"{image.name}_rough_segmented", labels, False)
    labels = smooth_labels(labels)
    yield (f"{image.name}_segmented", labels, False)


def blur_and_gray_image(
    image: "napari.types.ImageData", sigma: float = 2.0
) -> "napari.types.ImageData":
    image = np.asarray(image)
    if len(image.shape) >= 3 and image.shape[-1] in [3, 4]:
        image = color.rgb2gray(image)
    return filters.gaussian(image, sigma)


def initial_segment(
    image: "napari.types.ImageData", classes: int = 3
) -> "napari.types.LabelsData":
    image = np.asarray(image)
    thresholds = filters.threshold_multiotsu(image, classes=classes)
    return np.digitize(image, thresholds)


def smooth_labels(
    labels: "napari.types.LabelsData",
    smooth_threshold: int = 20,
    disk_size: int = 4,
):
    labels = np.asarray(labels)
    smoother_labels = filters.rank.mean(labels, morphology.disk(disk_size))

    return smoother_labels


@magic_factory
def segmentation_widget(viewer: "napari.Viewer", image: "napari.layers.Image"):
    def display_layer(args):
        name, data, as_image = args
        print(f"adding {data} to viewer")
        if name in viewer.layers:
            viewer.layers[name].data = data
        else:
            added = (
                viewer.add_image(data, name=name)
                if as_image
                else viewer.add_labels(data, name=name)
            )

    worker = segment(image)
    worker.yielded.connect(display_layer)
    worker.start()
