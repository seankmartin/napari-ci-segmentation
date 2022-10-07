from napari import Viewer, run, imshow
from PIL import Image
import numpy as np

image = Image.open("example_ci.png")
viewer, _ = imshow(np.asarray(image), name="CI")
viewer.window.add_plugin_dock_widget("napari-ci-segmentation", "CI Segmentation")
run()
