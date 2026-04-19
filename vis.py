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

state.volumeName = str(44)
state.volumeOptions = [{"title": str(n), "value": str(n)} for n in sorted(dataset.volumeNames)]

# TODO: Actual names of slices
state.sliceOption = str(0)
state.sliceOptions = [{"title": str(n), "value": str(n)} for n in range(4)]

state.overlayMode = "saliency"  # default
state.overlayOptions = [
    {"title": "Tumor Mask", "value": "tumor"},
    {"title": "Saliency", "value": "saliency"},
]

pv.set_plot_theme('dark')
pl = pv.Plotter()


def drawTumorGrid(tumorVolume):
    tumorGrid = pv.ImageData(dimensions=tumorVolume.shape)
    tumorGrid.point_data["tumor"] = tumorVolume.flatten(order="F")

    tumorLabels = ["Necrotic/Non-enhancing Tumor", "Peritumoral Edema", "GD-enhancing Tumor"]
    tumorColors = ["red", "blue", "orange"]

    legend = []

    for i in range(3):
        region = tumorGrid.threshold((i + 1, i + 1))

        if region.n_points > 0:
            pl.add_mesh(region, color=tumorColors[i], opacity=0.6)
            legend.append([tumorLabels[i], tumorColors[i]])

    pl.add_legend(
        legend,
        bcolor="black",
        border=True,
        size=(0.25, 0.25)
    )


def drawSaliency(volumeName):
    saliencyPath = os.path.join(config.saliencyDirectory, f"{volumeName}_saliency.npy")

    if os.path.exists(saliencyPath):
        saliency = np.load(saliencyPath)
        print(saliency.shape, np.min(saliency), np.max(saliency))

        saliency = np.transpose(saliency, (1, 2, 0))

        salGrid = pv.ImageData(dimensions=saliency.shape)
        salGrid.point_data["saliency"] = saliency.flatten(order="F")

        threshold = np.quantile(saliency, 0.8)
        region = salGrid.threshold((threshold, saliency.max()))

        if region.n_points > 0:
            pl.add_mesh(region, color="yellow", opacity=0.5)

            pl.add_legend(
                [["Saliency (Top 20%)", "yellow"]],
                bcolor="black",
                border=True,
                size=(0.25, 0.25)
            )


def loadVolume(volumeName, sliceOption, overlayMode):
    pl.clear()
    mriVolume, tumorVolume = dataset.loadVolume(volumeName)
    print(mriVolume.shape)
    mriVolume = mriVolume[:, int(sliceOption)].numpy()
    tumorVolume = tumorVolume.numpy()
    
    mriVolume = np.transpose(mriVolume, (1, 2, 0))
    tumorVolume = np.transpose(tumorVolume, (1, 2, 0))

    grid = pv.ImageData(dimensions=mriVolume.shape)
    grid.point_data["MRI"] = mriVolume.flatten(order="F")
    pl.add_volume(grid, cmap="bone_r")

    if overlayMode == "tumor":
        drawTumorGrid(tumorVolume)
    elif overlayMode == "saliency":
        drawSaliency(volumeName)

    pl.reset_camera()
    ctrl.view_update()  # tells Trame to push the new render to the browser


# Trame state change listener — fires whenever state.volume_name changes
@state.change("volumeName", "sliceOption", "overlayMode")
def onVolumeChange(volumeName, sliceOption, overlayMode, **kwargs):
    loadVolume(volumeName, sliceOption, overlayMode)


with SinglePageLayout(server, dark=True) as layout:
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

        v3.VSelect(
            label="MRI Slice",
            v_model=("sliceOption",),
            items=("sliceOptions",),
            density="compact",
            hide_details=True,
            style="max-width: 300px;",
        )

        v3.VSelect(
            label="Overlay",
            v_model=("overlayMode",),
            items=("overlayOptions",),
            density="compact",
            hide_details=True,
            style="max-width: 200px;",
        )

    with layout.content:
        with plotter_ui(pl) as view:
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            

server.start()