import os
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3
from trame.widgets import vtk as vtk_widgets
from utils import *

# TODO: Make config selection dynamic to allow for choosing where to load MRI data from
config = Config().load(os.path.join("configs", "config.json"))
dataset = BraTSData(config)

server = get_server()
state, ctrl = server.state, server.controller

# TODO: Add MRI slice selection as well as tumor slice selection
state.volumeName = str(dataset.volumeNames[0])
state.volumeOptions = [{"title": str(n), "value": str(n)} for n in dataset.volumeNames]

pl = pv.Plotter()


def loadVolume(volume_name):
    pl.clear()
    mriVolume, tumorVolume = dataset.loadVolume(volume_name)
    mriVolume = mriVolume[:, 0].numpy()
    tumorVolume = tumorVolume.numpy()

    grid = pv.ImageData(dimensions=mriVolume.shape)
    grid.point_data["MRI"] = mriVolume.flatten(order="F")
    pl.add_volume(grid, cmap="hot")

    # tumor_grid = pv.ImageData(dimensions=tumorVolume.shape)
    # tumor_grid.point_data["mask"] = tumorVolume.flatten(order="F")
    # pl.add_volume(tumor_grid, cmap="hot", opacity="linear", clim=[0.5, 1.0])

    pl.reset_camera()
    ctrl.view_update()  # tells Trame to push the new render to the browser


# Trame state change listener — fires whenever state.volume_name changes
@state.change("volumeName")
def on_volume_change(volumeName, **kwargs):
    loadVolume(volumeName)


with SinglePageLayout(server) as layout:
    layout.title.set_text("MRI Viewer")

    with layout.toolbar:
        v3.VSelect(
            label="Volume",
            v_model=("volumeName",),      # binds to state.volume_name
            items=("volumeOptions",),      # binds to state.volume_options
            density="compact",
            hide_details=True,
            style="max-width: 300px;",
        )

    with layout.content:
        with plotter_ui(pl):
            view = plotter_ui(pl)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            

server.start()