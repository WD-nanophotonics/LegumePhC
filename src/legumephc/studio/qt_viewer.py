from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from matplotlib import colormaps
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .inspection import InspectionRow, RecordView
from .plotting import sample_cell_polygons


class ResultCanvas(QtWidgets.QWidget):
    """Fast interactive rendering of immutable record samples."""

    rowSelected = QtCore.Signal(int)
    pinsChanged = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.plot = self.graphics.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setMenuEnabled(True)
        self.view: RecordView | None = None
        self.style: dict[str, Any] = {}
        self._hit_items: list[pg.ScatterPlotItem] = []
        self._layer_items: dict[str, list[Any]] = defaultdict(list)
        self._highlight = pg.ScatterPlotItem(size=13, symbol="o", pen=pg.mkPen("#111111", width=2), brush=None)
        self._pin_items: list[tuple[int, Any, Any]] = []
        self._colorbar = None
        self._interpolated = False
        self.plot.addItem(self._highlight)

    def clear_record(self) -> None:
        self.graphics.clear()
        self.plot = self.graphics.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self._hit_items.clear()
        self._layer_items.clear()
        self._pin_items.clear()
        self._colorbar = None
        self._highlight = pg.ScatterPlotItem(size=13, symbol="o", pen=pg.mkPen("#111111", width=2), brush=None)
        self.plot.addItem(self._highlight)

    def set_record(self, view: RecordView, style: dict[str, Any], *, pins: list[int] | None = None) -> None:
        self.clear_record()
        self.view = view
        self.style = dict(style)
        self._interpolated = bool(style.get("berry_render_mode") == "linear_interpolation")
        self.plot.showGrid(x=bool(style.get("grid", True)), y=bool(style.get("grid", True)), alpha=0.25)
        if style.get("legend", True) and view.operation in {"band_structure", "efs", "berry_curvature_dipole"}:
            self.plot.addLegend()
        renderer = getattr(self, f"_render_{view.operation}", self._render_points)
        renderer()
        self._add_hit_target()
        for index in pins or []:
            if 0 <= int(index) < len(view.rows):
                self.pin_row(int(index), emit=False)
        self.plot.autoRange()

    def _add_hit_target(self) -> None:
        if self.view is None or not self.view.rows:
            return
        positions = np.asarray([(row.x, row.y) for row in self.view.rows], dtype=float)
        hit = pg.ScatterPlotItem(
            pos=positions, data=np.arange(len(positions)), size=20, pxMode=True,
            pen=None, brush=pg.mkBrush(0, 0, 0, 0), hoverable=True, tip=None,
        )
        hit.sigHovered.connect(self._hovered)
        hit.sigClicked.connect(self._clicked)
        self.plot.addItem(hit)
        self._hit_items.append(hit)

    def _hovered(self, _item, points, _event) -> None:
        if not points or self.view is None:
            QtWidgets.QToolTip.hideText()
            return
        index = int(points[0].data())
        prefix = "Display interpolation — nearest raw sample\n" if self._interpolated else ""
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), prefix + self.view.rows[index].tooltip())
        self.rowSelected.emit(index)

    def _clicked(self, _item, points, _event) -> None:
        if points:
            self.pin_row(int(points[0].data()))

    def select_row(self, index: int) -> None:
        if self.view is None or not (0 <= index < len(self.view.rows)):
            self._highlight.clear()
            return
        row = self.view.rows[index]
        self._highlight.setData(pos=np.asarray([[row.x, row.y]]))

    def pin_row(self, index: int, *, emit: bool = True) -> None:
        if self.view is None or not (0 <= index < len(self.view.rows)):
            return
        existing = next((item for item in self._pin_items if item[0] == index), None)
        if existing is not None:
            self.plot.removeItem(existing[1])
            self.plot.removeItem(existing[2])
            self._pin_items.remove(existing)
        else:
            row = self.view.rows[index]
            marker = pg.ScatterPlotItem(pos=np.asarray([[row.x, row.y]]), size=10, symbol="x", pen=pg.mkPen("#111111", width=2))
            text = pg.TextItem(row.tooltip(), anchor=(0, 1), color="#111111", fill=pg.mkBrush(255, 255, 255, 220), border=pg.mkPen("#777777"))
            text.setPos(row.x, row.y)
            self.plot.addItem(marker)
            self.plot.addItem(text)
            self._pin_items.append((index, marker, text))
        if emit:
            self.pinsChanged.emit(self.pinned_rows())

    def pinned_rows(self) -> list[int]:
        return sorted(item[0] for item in self._pin_items)

    def clear_pins(self) -> None:
        for _, marker, text in self._pin_items:
            self.plot.removeItem(marker)
            self.plot.removeItem(text)
        self._pin_items.clear()
        self.pinsChanged.emit([])

    def set_layer_visible(self, name: str, visible: bool) -> None:
        for item in self._layer_items.get(name, []):
            item.setVisible(bool(visible))

    def layer_names(self) -> list[str]:
        return list(self._layer_items)

    def view_limits(self) -> tuple[list[float], list[float]]:
        (xmin, xmax), (ymin, ymax) = self.plot.viewRange()
        return [float(xmin), float(xmax)], [float(ymin), float(ymax)]

    def _render_band_structure(self) -> None:
        assert self.view is not None
        grouped: dict[int, list[InspectionRow]] = defaultdict(list)
        for row in self.view.rows:
            grouped[int(row.values["Band"])].append(row)
        line = bool(self.style.get("band_line", True))
        markers = bool(self.style.get("band_markers", False))
        for band, rows in sorted(grouped.items()):
            x = np.asarray([row.x for row in rows])
            y = np.asarray([row.y for row in rows])
            item = self.plot.plot(
                x, y, name=f"Band {band}",
                pen=pg.mkPen(pg.intColor(band - 1), width=float(self.style.get("linewidth", 1.5))) if line else None,
                symbol="o" if markers else None, symbolSize=float(self.style.get("marker_size", 4.0)),
                symbolBrush=pg.mkBrush(pg.intColor(band - 1)), symbolPen=None,
            )
            self._layer_items[f"Band {band}"].append(item)
        labels = self.view.summary.get("path_labels") or []
        if labels and grouped:
            count = len(next(iter(grouped.values())))
            ticks = []
            for position, label in zip(np.linspace(0, count - 1, len(labels)), labels):
                ticks.append((float(position), "Γ" if str(label).lower() in {"gamma", "g", "γ"} else str(label)))
            self.plot.getAxis("bottom").setTicks([ticks])
        self.plot.setLabel("left", next((key for key in self.view.columns if "frequency" in key.lower()), "Frequency"))
        self.plot.setLabel("bottom", "")

    def _render_berry(self) -> None:
        assert self.view is not None
        points = np.asarray([(row.x, row.y) for row in self.view.rows])
        values = np.asarray([float(row.values["Berry curvature"]) for row in self.view.rows])
        finite = values[np.isfinite(values)]
        limit = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-15) if len(finite) else 1.0
        low = float(self.style.get("berry_vmin")) if self.style.get("berry_vmin") is not None else -limit
        high = float(self.style.get("berry_vmax")) if self.style.get("berry_vmax") is not None else limit
        cmap_name = str(self.style.get("cmap", "RdBu_r"))
        mpl_cmap = colormaps.get_cmap(cmap_name)
        if self._interpolated and len(points) >= 3:
            from scipy.interpolate import griddata
            x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 160)
            y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 160)
            xx, yy = np.meshgrid(x, y)
            zz = griddata(points, values, (xx, yy), method="linear")
            image = pg.ImageItem(zz.T)
            image.setRect(QtCore.QRectF(float(x[0]), float(y[0]), float(x[-1] - x[0]), float(y[-1] - y[0])))
            image.setLookupTable((mpl_cmap(np.linspace(0, 1, 256)) * 255).astype(np.ubyte))
            image.setLevels((low, high))
            self.plot.addItem(image)
            self._layer_items["Berry map"].append(image)
        else:
            domain = np.asarray(self.view.arrays.get("domain_outline")) if "domain_outline" in self.view.arrays else None
            if domain is None:
                margin = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-3) * 0.05
                lower, upper = np.min(points, axis=0) - margin, np.max(points, axis=0) + margin
                domain = np.asarray([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
            for polygon, value in zip(sample_cell_polygons(points, domain), values):
                path = QtGui.QPainterPath(QtCore.QPointF(*polygon[0]))
                for vertex in polygon[1:]:
                    path.lineTo(QtCore.QPointF(*vertex))
                path.closeSubpath()
                item = QtWidgets.QGraphicsPathItem(path)
                fraction = float(np.clip((value - low) / (high - low), 0, 1))
                rgba = mpl_cmap(fraction)
                item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
                item.setBrush(QtGui.QBrush(QtGui.QColor.fromRgbF(*rgba)))
                self.plot.addItem(item)
                self._layer_items["Berry map"].append(item)
        invalid = [index for index, row in enumerate(self.view.rows) if row.values.get("Qualified") is False]
        if invalid:
            bad = pg.ScatterPlotItem(pos=points[invalid], symbol="x", size=12, pen=pg.mkPen("k", width=2))
            self.plot.addItem(bad)
            self._layer_items["Unqualified"].append(bad)
        if self.style.get("show_sample_centers", False):
            samples = pg.ScatterPlotItem(pos=points, symbol="o", size=4, pen=pg.mkPen("k"), brush=None)
            self.plot.addItem(samples)
            self._layer_items["Sample centers"].append(samples)
        if self.style.get("colorbar", True):
            color_map = pg.colormap.getFromMatplotlib(cmap_name)
            self._colorbar = pg.ColorBarItem(values=(low, high), colorMap=color_map, label="Berry curvature", interactive=False, width=16)
            self.graphics.addItem(self._colorbar, row=0, col=1)
            self._layer_items["Colorbar"].append(self._colorbar)
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "qₓ")
        self.plot.setLabel("left", "qᵧ")

    def _render_efs(self) -> None:
        assert self.view is not None
        grid_points = self.view.arrays.get("grid_qpoints")
        grid_shape = self.view.summary.get("grid_shape")
        mask = self.view.arrays.get("inside_bz_mask")
        if grid_points is not None and grid_shape and mask is not None:
            grid = np.asarray(grid_points).reshape(*grid_shape, 2)
            frequencies = np.asarray(self.view.arrays["frequencies"], dtype=float)
            unit_key = next((key for key in self.view.rows[0].values if "frequency" in key.lower()), None) if self.view.rows else None
            raw_key = next((key for key in self.view.rows[0].values if key == unit_key), None) if unit_key else None
            factor = (float(self.view.rows[0].values[raw_key]) / float(frequencies[0, 0])) if raw_key and frequencies[0, 0] != 0 else 1.0
            selected = int(self.style.get("component_index", 0)); selected = max(0, min(selected, frequencies.shape[1] - 1))
            values = np.full(tuple(grid_shape), np.nan); values.reshape(-1)[np.asarray(mask, dtype=bool).reshape(-1)] = frequencies[:, selected] * factor
            finite = values[np.isfinite(values)]
            if len(finite):
                levels = self.style.get("efs_levels") or np.linspace(float(np.min(finite)), float(np.max(finite)), 12)[1:-1]
                x0, y0 = float(grid[0, 0, 0]), float(grid[0, 0, 1]); dx = float(grid[0, 1, 0] - x0); dy = float(grid[1, 0, 1] - y0)
                cmap = pg.colormap.getFromMatplotlib(str(self.style.get("cmap", "viridis")))
                for index, level in enumerate(levels):
                    item = pg.IsocurveItem(data=values.T, level=float(level), pen=pg.mkPen(cmap.map(index / max(len(levels) - 1, 1), mode="qcolor"), width=1.5))
                    transform = QtGui.QTransform(); transform.translate(x0, y0); transform.scale(dx, dy); item.setTransform(transform)
                    self.plot.addItem(item); self._layer_items["Contours"].append(item)
                if self.style.get("colorbar", True):
                    self._colorbar = pg.ColorBarItem(values=(float(np.min(finite)), float(np.max(finite))), colorMap=cmap, label=unit_key or "Frequency", interactive=False, width=16)
                    self.graphics.addItem(self._colorbar, row=0, col=1); self._layer_items["Colorbar"].append(self._colorbar)
        grouped: dict[int, list[InspectionRow]] = defaultdict(list)
        for row in self.view.rows:
            grouped[int(row.values["Band"])].append(row)
        for band, rows in grouped.items():
            values = np.asarray([next(value for key, value in row.values.items() if "frequency" in key.lower()) for row in rows])
            finite = values[np.isfinite(values)]
            low, high = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (0.0, 1.0)
            brushes = [pg.mkBrush(pg.intColor(int(255 * (value - low) / max(high - low, 1e-15)), hues=256)) for value in values]
            item = pg.ScatterPlotItem(pos=np.asarray([(row.x, row.y) for row in rows]), size=7, brush=brushes, pen=None)
            self.plot.addItem(item)
            self._layer_items[f"Band {band}"].append(item)
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "qₓ")
        self.plot.setLabel("left", "qᵧ")

    def _render_fields_energy(self) -> None:
        assert self.view is not None
        quantity = str(self.style.get("field_quantity", "energy_density"))
        values = np.asarray(self.view.arrays.get(quantity, self.view.arrays["energy_density"]), dtype=float)
        component = max(0, min(int(self.style.get("component_index", 0)), values.shape[-1] - 1))
        image_values = values[0, :, :, component]
        if "display_x_corners" in self.view.arrays and "display_y_corners" in self.view.arrays:
            x = self.view.arrays["display_x_corners"]; y = self.view.arrays["display_y_corners"]
            image = pg.PColorMeshItem(x, y, image_values, colorMap=pg.colormap.getFromMatplotlib(str(self.style.get("cmap", "viridis"))))
        else:
            image = pg.ImageItem(image_values.T)
        self.plot.addItem(image); self._layer_items[quantity].append(image)
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "x index")
        self.plot.setLabel("left", "y index")

    def _render_frequency_at_k(self) -> None:
        assert self.view is not None
        if self.view.rows:
            row = self.view.rows[0]
            bar = pg.BarGraphItem(x=[0], height=[row.y], width=0.6, brush=pg.mkBrush("#377eb8"))
            self.plot.addItem(bar)
            self._layer_items["Frequency"].append(bar)
            self.plot.setLabel("left", next((key for key in row.values if "frequency" in key.lower()), "Frequency"))

    def _render_berry_curvature_dipole(self) -> None:
        self._render_points()
        if self.view is not None and "first_moment" in self.view.arrays:
            vector = np.asarray(self.view.arrays["first_moment"]).reshape(2)
            arrow = pg.ArrowItem(pos=(vector[0], vector[1]), angle=float(np.degrees(np.arctan2(vector[1], vector[0]))), headLen=15)
            self.plot.addItem(arrow)
            self._layer_items["First moment"].append(arrow)

    def _render_points(self) -> None:
        if self.view is None or not self.view.rows:
            return
        item = pg.ScatterPlotItem(pos=np.asarray([(row.x, row.y) for row in self.view.rows]), size=7, brush=pg.mkBrush("#377eb8"), pen=None)
        self.plot.addItem(item)
        self._layer_items["Samples"].append(item)
