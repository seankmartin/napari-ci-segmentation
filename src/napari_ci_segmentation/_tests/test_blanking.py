import napari
import numpy as np
from pytest import mark
from magicgui import magic_factory

@magic_factory(auto_call=False, call_button=True)
def my_widget(image : "napari.layers.Image") -> napari.types.LayerDataTuple:
    zero_image = np.zeros(image.data.shape)
    return zero_image, {"name": f"zero {image.name}"}, "image"
    
	
@mark.parametrize("data_shape", [(100, 100), (100, 100, 100)])
def test_my_widget(make_napari_viewer, data_shape):
    # 1. Setup viewer and widget, load data etc.
    viewer = make_napari_viewer()
    wdg = my_widget()
    viewer.window.add_dock_widget(wdg)
    data = np.random.randint(0, 255, size=data_shape)

    # 2. Perform actions and assert they work as expected
    layer = viewer.add_image(data)
    assert wdg.image.value == layer
    wdg()
    result = viewer.layers[-1].data
    assert np.all(result == 0)
    assert result.shape == data_shape

if __name__ == '__main__':
    test_my_widget(napari.Viewer, (200, 200))
    napari.run()
