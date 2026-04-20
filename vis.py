import os
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, html
from trame.widgets import vtk as vtk_widgets
from utils import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64, io

# TODO: Make config selection dynamic to allow for choosing where to load MRI data from
config = Config().load(os.path.join("configs", "resnetConfig.json"))
dataset = BraTSData(config)

server = get_server()
state, ctrl = server.state, server.controller

state.volumeName = str(369)
state.volumeOptions = [{"title": str(n), "value": str(n)} for n in sorted(dataset.volumeNames)]

# TODO: Actual names of slices
state.sliceOption = str(0)
state.sliceOptions = [{"title": str(n), "value": str(n)} for n in range(4)]

state.overlayMode = "saliency"  # default
state.overlayOptions = [
    {"title": "Tumor Mask", "value": "tumor"},
    {"title": "Saliency", "value": "saliency"},
]

# 2D slice index along the Z axis; discovered dynamically per volume
state.sliceIndex = 83
state.sliceIndexMin = 0
state.sliceIndexMax = 0   # updated once a volume is loaded
state.sliceAvailable = False  # whether an overlay file exists for current sliceIndex
state.sliceImageSrc = ""      # base-64 PNG shown in the right pane

pv.set_plot_theme('dark')
pl = pv.Plotter()

_cache = {"mriVolume": None, "tumorVolume": None, "volumeName": None}


def _load_saliency(volumeName, sliceOption, sliceIndex=None):
    if sliceIndex is not None:
        path = os.path.join(
            config.sampleDirectory,
            f"{volumeName}_{sliceIndex}_saliency.npy",
        )
        if os.path.exists(path):
            return np.load(path)
        return None  # not available for this Z slice
    else:
        # fallback: try modality-based name used by 3-D view
        path = os.path.join(config.sampleDirectory, f"{volumeName}_saliency.npy")
        if os.path.exists(path):
            return np.load(path)
        return None
    

def _render_2d_slice(volumeName, sliceOption, sliceIndex, overlayMode):
    """
    Build a base-64 PNG of the 2-D MRI slice with optional overlay.
    Returns (img_src: str, available: bool).
    """
    if _cache["mriVolume"] is None or _cache["volumeName"] != volumeName:
        return "", False

    mriVolume = _cache["mriVolume"]   # shape (D, C, H, W)  — raw tensor
    depth = mriVolume.shape[0]
    zi = max(0, min(int(sliceIndex), depth - 1))

    # MRI slice: (H, W)
    mri2d = mriVolume[zi, int(sliceOption)].numpy()

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(mri2d, cmap="bone_r", origin="lower", aspect="equal")
    ax.axis("off")

    overlay_found = False

    if overlayMode == "tumor":
        tumorVolume = _cache["tumorVolume"]  # (H, W, D)
        tumor2d = tumorVolume[:, :, zi]
        colors_map = {1: ([1, 0, 0], "Necrotic"), 2: ([0, 0, 1], "Edema"), 3: ([1, 0.5, 0], "Enhancing")}
        for label, (rgb, _name) in colors_map.items():
            mask = (tumor2d == label)
            if mask.any():
                rgba = np.zeros((*mask.shape, 4))
                rgba[mask] = [*rgb, 0.55]
                ax.imshow(rgba, origin="lower", aspect="equal")
                overlay_found = True

    elif overlayMode == "saliency":
        sal = _load_saliency(volumeName, sliceOption, zi)
        if sal is not None:
            # sal shape may be (H, W) or (1, H, W) — normalise
            if sal.ndim == 3:
                sal = sal[0]
            sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            threshold = np.quantile(sal_norm, 0.9)
            mask = sal_norm >= threshold
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask, 0] = 1.0   # yellow R
            rgba[mask, 1] = 1.0   # yellow G
            rgba[mask, 3] = 0.55
            ax.imshow(rgba, origin="lower", aspect="equal")
            overlay_found = True

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="black", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}", overlay_found


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
    saliency = _load_saliency(volumeName, state.sliceOption)
    if saliency is None:
        return

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
    mriSlice = mriVolume[:, int(sliceOption)].numpy()
    tumorVolume = tumorVolume.numpy()
    
    mriSlice = np.transpose(mriSlice, (1, 2, 0))
    tumorVolume = np.transpose(tumorVolume, (1, 2, 0))

    # Cache for the 2-D pane
    _cache["mriVolume"]  = mriVolume
    _cache["tumorVolume"] = tumorVolume
    _cache["volumeName"]  = volumeName

    depth = mriSlice.shape[2]
    state.sliceIndexMax = depth - 1
    state.sliceIndex = min(state.sliceIndex, depth - 1)

    grid = pv.ImageData(dimensions=mriSlice.shape)
    grid.point_data["MRI"] = mriSlice.flatten(order="F")
    pl.add_volume(grid, cmap="bone_r")

    if overlayMode == "tumor":
        drawTumorGrid(tumorVolume)
    elif overlayMode == "saliency":
        drawSaliency(volumeName)

    pl.reset_camera()
    ctrl.view_update()  # tells Trame to push the new render to the browser



@state.change("volumeName", "sliceOption", "overlayMode")
def onVolumeChange(volumeName, sliceOption, overlayMode, **kwargs):
    loadVolume(volumeName, sliceOption, overlayMode)
    img_src, available = _render_2d_slice(volumeName, sliceOption, state.sliceIndex, overlayMode)
    state.sliceImageSrc  = img_src
    state.sliceAvailable = available


@state.change("sliceIndex")
def onSliceIndexChange(sliceIndex, **kwargs):
    img_src, available = _render_2d_slice(
        state.volumeName, state.sliceOption, sliceIndex, state.overlayMode
    )
    state.sliceImageSrc  = img_src
    state.sliceAvailable = available


with SinglePageLayout(server, dark=True) as layout:
    layout.title.set_text("MRI Viewer")

    with layout.toolbar:
        v3.VSelect(
            label="Volume",
            v_model=("volumeName",),
            items=("volumeOptions",),
            density="compact",
            hide_details=True,
            style="max-width: 300px;",
        )

        v3.VSelect(
            label="MRI Modality",
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
        with v3.VRow(no_gutters=True, style="height: 80%; flex-wrap: nowrap;"):

            # Left pane — 3-D volume
            with v3.VCol(style="height: 100%; min-width: 0;"):
                with plotter_ui(pl) as view:
                    ctrl.view_update       = view.update
                    ctrl.view_reset_camera = view.reset_camera

            # Right pane — 2-D slice + slider
            with v3.VCol(
                style=(
                    "height: 100%; display: flex; flex-direction: column; "
                    "align-items: center; background: #111; min-width: 0;"
                )
            ):
                # Slice image
                html.Img(
                    src=("sliceImageSrc",),
                    style=(
                        "flex: 1; object-fit: contain; width: 100%; "
                        "max-height: calc(100% - 90px);"
                    ),
                )

                # "No overlay available" chip
                v3.VChip(
                    "No overlay for this slice",
                    color="warning",
                    size="small",
                    style="margin-top: 6px;",
                    v_show="!sliceAvailable && overlayMode !== 'none'",
                )

                # Slider
                with html.Div(style="width: 90%; padding: 8px 0;"):
                    html.Div(
                        "{{ 'Z Slice: ' + sliceIndex }}",
                        style="color: #ccc; font-size: 12px; text-align: center; margin-bottom: 4px;",
                    )
                    v3.VSlider(
                        v_model=("sliceIndex",),
                        min=("sliceIndexMin",),
                        max=("sliceIndexMax",),
                        step=1,
                        thumb_label=True,
                        color="primary",
                        track_color="grey",
                        hide_details=True,
                    )       

server.start()