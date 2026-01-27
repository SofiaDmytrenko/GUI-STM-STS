from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QTabWidget, QSplitter, QFileDialog, QMessageBox,
    QComboBox, QToolButton, QMenu, QCheckBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import csv
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
from gui.helpers.metadata_tab import MetadataTab
from gui.helpers.styles import Style


logger = logging.getLogger(__name__)

class TopoViewer(QMainWindow):
    DEFAULT_CMAPS = [
        "viridis", "terrain", "plasma", "inferno", "magma", "cividis",
        "gray", "binary", "Purples", "BuPu", "YlOrRd",
        "RdBu", "coolwarm", "twilight", "turbo",
        "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"
    ]

    def __init__(self, channel_data: dict, parent=None):
        super().__init__(parent)

        self.channel_data = channel_data
        self.filename = channel_data.get("attrs", {}).get("filename", "Unknown")

        self.resize(900, 600)
        self.setMinimumSize(700, 400)

        # --- Toolbar ---
        self.btn_height = 32
        self.create_widgets()
        self.style_widgets()
        self.create_layout()

        # ---------------- Data ----------------
        self.original_data = None
        self.display_data = None
        self.load_data()

        # ---------------- Central widget ----------------
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ---------------- Tabs ----------------
        tabs = QTabWidget()

        # Plot tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        # Splitter: left=canvas, right=info panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setStyleSheet(Style.SPLITTER)
        splitter.setChildrenCollapsible(False)

        # Matplotlib figure & canvas
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        splitter.addWidget(self.canvas)

        # Freeze layout once
        self.figure.subplots_adjust(right=0.9)

        # --- Right-side info panel ---
        info_panel = QWidget()
        info_panel.setObjectName("infoPanel")
        info_panel.setStyleSheet(Style.PANEL)
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(6, 6, 6, 6)

        # Section title
        info_label = QLabel("| INFO PANEL:")
        Style.style_title(info_label)
        info_layout.addWidget(info_label)

        # Title label
        channel_label = self.channel_data.get("title", "")
        self.lbl_title = QLabel(f"🏷️ {channel_label}")
        self.lbl_title.setToolTip("Channel label")
        self.lbl_title.setStyleSheet(Style.TOPO_LABEL)
        info_layout.addWidget(self.lbl_title)

        # Cursor block
        cursor_container = QWidget()
        cursor_container.setStyleSheet(Style.TOPO_LABEL_CONTAINER)
        cursor_layout = QVBoxLayout(cursor_container)

        self.pos_label = QLabel("📍 Cursor coordinates:")
        self.pos_label.setToolTip("Cursor position in real-world units")
        self.pos_label.setStyleSheet("QLabel { font-size: 13px; padding: 0; }")

        self.pos_xyz_label = QLabel("X: -,\nY: -,\nZ: -")
        self.pos_xyz_label.setStyleSheet("QLabel { padding-left: 26px; }")

        cursor_layout.addWidget(self.pos_label)
        cursor_layout.addWidget(self.pos_xyz_label)
        info_layout.addWidget(cursor_container)

        info_layout.addStretch(1)

        # --- Wrapper to add spacing between splitter and right panel ---
        self.right_wrapper = QWidget()
        wrapper_layout = QVBoxLayout(self.right_wrapper)
        wrapper_layout.setContentsMargins(6, 0, 0, 0)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(info_panel)

        # --- Add widgets to splitter ---
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.right_wrapper)

        # Set stretch factors so canvas takes more space than right panel
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        # ---------------- Internal state ----------------
        self.im = None
        self.colorbar = None

        # Initialize plot
        self.plot_topography()
        self.canvas.draw_idle()
        self._base_ax_pos = self.ax.get_position()

        # Mouse move
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        # ------------------- Add widgets to main layout -------------------
        # Tabs
        layout.addWidget(tabs)

        plot_layout.addWidget(splitter)  # Splitter containing canvas and info panel
        plot_layout.insertWidget(0, self.frame)
        tabs.addTab(plot_tab, "🗺️ TOPOGRAPHY")

        self.metadata_tab = MetadataTab(channel_data=self.channel_data)
        self.metadata_tab.set_metadata(self.channel_data.get("attrs", {}))
        tabs.addTab(self.metadata_tab, "ℹ️ METADATA")
        tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------- Data loading / processing -----------------
    def load_data(self):
        data = self.channel_data.get("data")
        if data is None:
            self.original_data = None
            self.display_data = None
            return
        arr = np.array(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("TopoViewer expects 2D array-like data")
        self.original_data = arr
        self.display_data = arr.copy()

    # ------------------------- Plotting ---------------------------------
    def plot_topography(self):
        if self.display_data is None and self.original_data is None:
            return

        # Use display_data if available, otherwise fallback to original
        data = self.display_data if self.display_data is not None else self.original_data

        fig = self.canvas.figure

        # Clear main axes content but keep the axes object
        self.ax.clear()

        # Remove previous dedicated cax if exists
        if getattr(self, "_cax", None) is not None:
            try:
                self._cax.remove()
            except Exception as e:
                logger.warning("Failed to remove previous colorbar axes: %s", e)
            self._cax = None
            self.colorbar = None

        # Draw image on main axes
        self.im = self.ax.imshow(
            data,
            cmap=self.cmap_combo.currentText(),
            origin="lower",
            aspect="equal"
        )

        # # Ensure color scaling matches the actual data range
        # try:
        #     vmin = float(np.nanmin(data))
        #     vmax = float(np.nanmax(data))
        #     if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
        #         self.im.set_clim(vmin, vmax)
        # except Exception as e:
        #     logger.warning("Failed to set exact color scaling", e)

        # Create a dedicated, fixed-size colorbar axes to the right
        divider = make_axes_locatable(self.ax)
        self._cax = divider.append_axes("right", size="4%", pad=0.08)

        # Create colorbar in the dedicated axes
        self.colorbar = fig.colorbar(self.im, cax=self._cax)
        self.update_axes_labels_and_ticks()
        self.ax.set_title(self.channel_data.get("title", "Topography"))

        self.canvas.draw_idle()

    # ------------------------- UI callbacks ------------------------------
    def change_cmap(self, cmap_name: str):
        if self.im is None:
            return
        self.im.set_cmap(cmap_name)
        self.canvas.draw()

    def smooth_box(self, data: np.ndarray, smooth_percent: int) -> np.ndarray:
        # Map percent (0-100) to internal smooth_value (0-10)
        smooth_value = int(round(smooth_percent * 10 / 100))
        if smooth_value <= 0:
            return data.copy()

        kernel = 1 + 2 * smooth_value  # odd kernel size
        pad = kernel // 2
        temp = np.empty_like(data)
        k = np.ones(kernel) / kernel

        # Convolve X
        for j in range(data.shape[0]):
            temp[j, :] = np.convolve(np.pad(data[j, :], pad_width=pad, mode='edge'), k, mode='valid')

        # Convolve Y
        out = np.empty_like(data)
        for i in range(data.shape[1]):
            out[:, i] = np.convolve(np.pad(temp[:, i], pad_width=pad, mode='edge'), k, mode='valid')
        return out

    # ------------------------- Save / Export -----------------------------
    def save_image(self):
        if self.im is None:
            QMessageBox.warning(self, "No image", "Nothing to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
        if not file_path:
            return
        # tighten layout so colorbar included
        self.canvas.figure.savefig(file_path, bbox_inches='tight', dpi=300)
        QMessageBox.information(self, "Saved", f"Image saved to:\n{file_path}")

    def export_csv(self):
        if self.display_data is None:
            QMessageBox.warning(self, "No data", "No data to export.")
            return

        # File dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Topography", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        # --- Safely remove existing file ---
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot overwrite existing file:\n{e}")
                return

        # Metadata source
        attrs = self.channel_data.get("attrs", {})

        # --- Compute X/Y axes in meters ---
        xscale = attrs.get("RHK_Xscale", None)
        xsize = attrs.get("RHK_Xsize", None)
        xunit = attrs.get("RHK_Xunits", "m")

        yscale = attrs.get("RHK_Yscale", None)
        ysize = attrs.get("RHK_Ysize", None)
        yunit = attrs.get("RHK_Yunits", "m")

        ny, nx = self.display_data.shape

        x_range = abs(xscale * xsize) if xscale is not None and xsize is not None else nx
        y_range = abs(yscale * ysize) if yscale is not None and ysize is not None else ny

        x_axis = np.linspace(0, x_range, nx)
        y_axis = np.linspace(0, y_range, ny)

        # --- Helper: sanitize metadata ---
        def sanitize_metadata_value(value):
            s = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
            return f'{s}'

        data = np.array(self.display_data)

        try:
            with open(file_path, "w", newline='', encoding="utf-8", errors="replace") as f:
                writer = csv.writer(f)

                # --- Metadata rows ---
                metadata_keys = ["Attribute"]
                metadata_values = ["Value"]

                # Define the preferred order
                preferred_order = [
                    "filename", "RHK_Label", "long_name", "RHK_SessionText", "RHK_Bias", "RHK_Ysize", "RHK_Xsize",
                    "RHK_Xlabel", "RHK_Ylabel", "RHK_Xscale", "RHK_Yscale",
                    "RHK_Xunits", "RHK_Yunits", "RHK_Zunits"
                ]

                # Add preferred keys first if they exist
                for key in preferred_order:
                    if key in attrs:
                        metadata_keys.append(sanitize_metadata_value(key))
                        metadata_values.append(sanitize_metadata_value(attrs[key]))

                # Add the rest of the keys
                for k, v in attrs.items():
                    if k in ["RHK_PRMdata"] or k in preferred_order:
                        continue
                    metadata_keys.append(sanitize_metadata_value(k))
                    metadata_values.append(sanitize_metadata_value(v))

                writer.writerow(metadata_keys)
                writer.writerow(metadata_values)
                writer.writerow([])  # blank line

                # --- Row headers for topography ---
                row_axis = [f"X [{xunit}]", f"Y [{yunit}]", "Index"] + list(range(nx))
                writer.writerow(row_axis)

                # --- Data rows: one row per Y ---
                for j in range(ny):
                    row = [x_axis[j % nx], y_axis[j], j]  # first 3 columns
                    row += [f"{data[j, i]:.16e}" for i in range(nx)]
                    writer.writerow(row)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export topography:\n{e}")
            return

        QMessageBox.information(self, "Exported", f"Topography saved to:\n{file_path}")

    # ------------------------- Utilities ---------------------------------
    def recompute_display(self):
        if self.original_data is None:
            return

        # --- Base data ---
        if hasattr(self, "flattened_data"):
            data = self.flattened_data.copy()
        else:
            data = self.original_data.copy()

        # --- Apply smoothing ---
        if getattr(self, "smooth_slider", None):
            smooth_value = self.smooth_slider.value()
            if smooth_value > 0:
                data = self.smooth_box(data, smooth_value)

        self.display_data = data

        # --- Update image ---
        if getattr(self, "im", None) is not None:
            self.im.set_data(self.display_data)

            # Auto-contrast
            if getattr(self, "auto_contrast_checkbox", None) and self.auto_contrast_checkbox.isChecked():
                self.apply_auto_contrast()
            else:
                if hasattr(self, "flattened_data"):
                    vmin = 0.0
                    vmax = np.nanmax(self.display_data)
                else:
                    vmin = np.nanmin(self.display_data)
                    vmax = np.nanmax(self.display_data)
                self.im.set_clim(vmin, vmax)

            # Update axes labels and ticks
            self.update_axes_labels_and_ticks()

            # Update colorbar
            if getattr(self, "colorbar", None) is not None:
                self.colorbar.update_normal(self.im)

            # Redraw
            if getattr(self, "canvas", None) is not None:
                self.canvas.draw_idle()
        else:
            self.plot_topography()

    def apply_auto_contrast(self):
        if self.display_data is None or self.im is None:
            return
        try:
            low = float(np.nanpercentile(self.display_data, 1.0))
            high = float(np.nanpercentile(self.display_data, 99.0))
            self.im.set_clim(low, high)
            if getattr(self, "colorbar", None) is not None:
                self.colorbar.update_normal(self.im)
            self.canvas.draw_idle()
        except Exception:
            # fallback: min/max
            vmin, vmax = np.nanmin(self.display_data), np.nanmax(self.display_data)
            self.im.set_clim(vmin, vmax)
            if getattr(self, "colorbar", None):
                self.colorbar.update_normal(self.im)
            self.canvas.draw_idle()

    def flatten_clicked(self):
        if self.original_data is None:
            return

        ny, nx = self.original_data.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        Z = self.original_data

        # Fit plane: Z = a*x + b*y + c
        A = np.column_stack((X.ravel(), Y.ravel(), np.ones(X.size)))
        coeff, _, _, _ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
        plane = coeff[0] * X + coeff[1] * Y + coeff[2]

        # Subtract plane
        flattened = Z - plane

        # Shift so min = 0
        flattened -= np.nanmin(flattened)

        # Store flattened data
        self.flattened_data = flattened

        # Recompute display (apply smoothing, auto-contrast)
        self.recompute_display()

    def reset_viewer(self):
        if self.original_data is None:
            return

        self.display_data = self.original_data.copy()
        if hasattr(self, "flattened_data"):
            del self.flattened_data

        # Reset smooth control
        if hasattr(self, "smooth_slider"):
            self.smooth_slider.setValue(0)

        # Reset auto-contrast checkbox
        if hasattr(self, "auto_contrast_checkbox"):
            self.auto_contrast_checkbox.setChecked(False)

        # Update image and colorbar
        if getattr(self, "im", None):
            self.im.set_data(self.display_data)
            self.im.set_clim(np.nanmin(self.display_data), np.nanmax(self.display_data))
            if getattr(self, "colorbar", None):
                self.colorbar.update_normal(self.im)
            if getattr(self, "canvas", None):
                self.canvas.draw_idle()

    def update_axes_labels_and_ticks(self):
        if self.display_data is None or self.ax is None:
            return

        attrs = self.channel_data.get("attrs", {})

        xscale = attrs.get("RHK_Xscale", None)
        xsize = attrs.get("RHK_Xsize", None)
        xunit = attrs.get("RHK_Xunits", "m")

        yscale = attrs.get("RHK_Yscale", None)
        ysize = attrs.get("RHK_Ysize", None)
        yunit = attrs.get("RHK_Yunits", "m")

        ny, nx = self.display_data.shape

        # Compute full range in meters
        x_range = abs(xscale * xsize) if xscale is not None and xsize is not None else nx
        y_range = abs(yscale * ysize) if yscale is not None and ysize is not None else ny

        # Auto-scale
        _, x_unit_disp = self.auto_scale(x_range, xunit)
        _, y_unit_disp = self.auto_scale(y_range, yunit)

        self.ax.set_xlabel(f"X [{x_unit_disp}]")
        self.ax.set_ylabel(f"Y [{y_unit_disp}]")

        # --- Set colorbar label ---
        if getattr(self, "colorbar", None) is not None:
            zunit = self.channel_data.get("attrs", {}).get("RHK_Zunits", "m")
            self.colorbar.set_label(f"Z [{zunit}]")

        # Coordinates for ticks
        if xscale is not None and xsize is not None:
            x_coords = np.linspace(0, x_range, nx)
        else:
            x_coords = np.arange(nx)

        if yscale is not None and ysize is not None:
            y_coords = np.linspace(0, y_range, ny)
        else:
            y_coords = np.arange(ny)

        num_ticks = 6
        xticks_idx = np.linspace(0, nx - 1, num=num_ticks, dtype=int)
        yticks_idx = np.linspace(0, ny - 1, num=num_ticks, dtype=int)

        # Scale tick coordinates for display
        x_ticks_display = [self.auto_scale(x_coords[i], xunit)[0] for i in xticks_idx]
        y_ticks_display = [self.auto_scale(y_coords[i], yunit)[0] for i in yticks_idx]

        self.ax.xaxis.set_major_locator(FixedLocator(xticks_idx))
        self.ax.xaxis.set_major_formatter(FixedFormatter([f"{val:.1f}" for val in x_ticks_display]))

        self.ax.yaxis.set_major_locator(FixedLocator(yticks_idx))
        self.ax.yaxis.set_major_formatter(FixedFormatter([f"{val:.1f}" for val in y_ticks_display]))

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or self.display_data is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)
        if x < 0 or y < 0 or y >= self.display_data.shape[0] or x >= self.display_data.shape[1]:
            return

        # --- Metadata for scaling ---
        attrs = self.channel_data.get("attrs", {})
        xscale = attrs.get("RHK_Xscale", 1.0)
        yscale = attrs.get("RHK_Yscale", 1.0)
        xunit = attrs.get("RHK_Xunits", "m")
        yunit = attrs.get("RHK_Yunits", "m")
        zunit = attrs.get("RHK_Zunits", "m")

        # Z value (use flattened if available)
        if hasattr(self, "flattened_data"):
            z_value = float(self.flattened_data[y, x])
        else:
            z_value = float(self.original_data[y, x])

        # Convert pixels to real coordinates (X, Y) with auto-scaling (X, Y, Z)
        x_real, xunit_disp = self.auto_scale(x * abs(xscale), xunit)
        y_real, yunit_disp = self.auto_scale(y * abs(yscale), yunit)
        z_real, zunit_disp = self.auto_scale(z_value, zunit)

        # --- Display formatted coordinates ---
        self.pos_label.setText(
            "📍 Cursor coordinates:")
        self.pos_xyz_label.setText(
            f"X = {x_real:.4f} {xunit_disp},\n"
            f"Y = {y_real:.4f} {yunit_disp},\n"
            f"Z = {z_real:.4f} {zunit_disp}."
        )

    def auto_scale(self, value, unit="m"):
        if unit.lower() != "m":
            return value, unit
        abs_val = abs(value)
        if abs_val >= 1e-3:
            return value * 1e3, "mm"
        elif abs_val >= 1e-6:
            return value * 1e6, "µm"
        elif abs_val >= 1e-9:
            return value * 1e9, "nm"
        else:
            return value * 1e12, "pm"

    # ----------------------------------------------------------
    #   Create widgets
    # ----------------------------------------------------------
    def create_widgets(self):
        # Save button
        self.btn_save = QToolButton()
        self.btn_save.setText("💾 Save")
        self.btn_save.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_save.clicked.connect(self.btn_save.showMenu)

        # Save menu
        self.save_menu = QMenu(self)
        self.save_menu.addAction("📈 Save chart (.png)", lambda: self.save_image())
        self.save_menu.addAction("📆 Export data (.csv)", lambda: self.export_csv())
        self.btn_save.setMenu(self.save_menu)

        # Colormap chooser
        self.cmap_label = QLabel("🎨 Colormap:")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.DEFAULT_CMAPS)

        self.cmap_combo.setCurrentText("viridis")

        # Flatten
        self.btn_flatten = QToolButton()
        self.btn_flatten.setText("📐 Fix Plane")

        # Smooth block
        self.smooth_label = QLabel("🖍️ Smooth:")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(0)
        self.smooth_value_label = QLabel(f"{self.smooth_slider.value()}%")

        # Layout for slider + value label
        self.slider_widget = QWidget()
        slider_layout = QHBoxLayout(self.slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(self.smooth_slider)
        slider_layout.addWidget(self.smooth_value_label)

        self.auto_contrast_checkbox = QCheckBox(" 🌓 Auto-contrast")

        self.btn_reset = QToolButton()
        self.btn_reset.setText("🔄 Reset")

        # Connect signals
        self.cmap_combo.currentTextChanged.connect(self.change_cmap)
        self.btn_flatten.clicked.connect(self.flatten_clicked)
        self.smooth_slider.valueChanged.connect(
            lambda val: self.smooth_value_label.setText(f"{val}%") or self.recompute_display())
        self.auto_contrast_checkbox.stateChanged.connect(self.recompute_display)
        self.btn_reset.clicked.connect(self.reset_viewer)

        # --- Tooltips ---
        self.btn_save.setToolTip("Save topography image or export data")
        self.cmap_label.setToolTip("Select colormap for topography")
        self.btn_flatten.setToolTip("Subtract best-fit plane and shift minimum to zero")
        self.smooth_label.setToolTip("Apply smoothing filter (0 = None)")
        self.auto_contrast_checkbox.setToolTip("Apply contrast using 1%-99% percentiles")
        self.btn_reset.setToolTip("Reset to original data")

    # ----------------------------------------------------------
    #   Style widgets
    # ----------------------------------------------------------
    def style_widgets(self):
        all_buttons = [
                self.btn_save,
                self.btn_flatten,
                self.btn_reset,
            ]
        Style.style_buttons(all_buttons)

        Style.style_cmap_combo(self.cmap_label, self.cmap_combo)
        Style.style_smooth_widgets(self.smooth_label, self.smooth_slider, self.smooth_value_label)
        Style.style_checkbox(self.auto_contrast_checkbox)
        Style.style_menu(self.save_menu)

    # ----------------------------------------------------------
    #   Layout
    # ----------------------------------------------------------
    def create_layout(self):
        layout, self.frame = Style.create_toolbar_frame()

        # --- Add buttons ---
        layout.addWidget(self.btn_save)
        layout.addWidget(Style.create_pixel_divider())
        layout.addWidget(self.cmap_label)
        layout.addWidget(self.cmap_combo)
        layout.addWidget(Style.create_pixel_divider())
        layout.addWidget(self.btn_flatten)
        layout.addWidget(Style.create_pixel_divider())
        layout.addWidget(self.smooth_label)
        layout.addWidget(self.smooth_slider)
        layout.addWidget(self.slider_widget)
        layout.addWidget(Style.create_pixel_divider())
        layout.addWidget(self.auto_contrast_checkbox)
        layout.addWidget(Style.create_pixel_divider())
        layout.addWidget(self.btn_reset)
