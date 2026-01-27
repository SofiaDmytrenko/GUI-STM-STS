from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QSplitter, QScrollArea, QFormLayout,
    QFileDialog, QMessageBox, QLineEdit, QDialog, QDialogButtonBox,
    QInputDialog, QTabWidget, QStyle, QStyleOptionViewItem
)
from PyQt6.QtGui import QRegularExpressionValidator, QCursor
from PyQt6.QtCore import Qt, QRegularExpression
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import os
import csv
import logging

from gui.channel_viewers.sts_viewer_folder.sts_toolbar import STSToolbar
from gui.channel_viewers.sts_viewer_folder.sts_processing import STSPlotter, STSOperations
from gui.channel_viewers.sts_viewer_folder.fit_gauss import FitGauss
from gui.helpers.metadata_tab import MetadataTab
from gui.helpers.styles import Style

logger = logging.getLogger(__name__)

class STSViewer(QMainWindow):
    def __init__(self, channel_data: dict, filename: str = None,
                 current_channel: dict = None, parent=None):
        super().__init__(parent)
        self.channel_data = channel_data
        self.filename = filename or channel_data.get("attrs", {}).get("filename", "Unknown")
        self.title = channel_data.get("title", "unknown")
        self.current_channel = current_channel

        # --- Initialize plot & data ---
        self.all_curves = []
        self.plotted_curves = []

        src_data = channel_data.get("data")
        self.original_data = src_data
        self.modified_data = np.asarray(src_data, dtype=float) if src_data is not None else None

        # Pre-populate raw curves
        if self.modified_data is not None and not self.all_curves:
            n_points, n_curves = self.modified_data.shape
            attrs = self.channel_data.get("attrs", {})
            x_offset = float(attrs.get("RHK_Bias", 0) or 0)
            x_scale = float(attrs.get("RHK_Xscale", 1) or 1)

            x_full = np.linspace(
                x_offset,
                x_offset + x_scale * (n_points - 1),
                n_points,
                dtype=float
            )

            self.all_curves = []
            csv_labels = attrs.get("CSV_Ylabels", [])

            for i in range(n_curves):
                y = np.asarray(self.modified_data[:, i], dtype=float)
                y = np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan)

                if i < len(csv_labels) and csv_labels[i]:
                    label = str(csv_labels[i])
                else:
                    label = f"C{i + 1}"

                self.all_curves.append({
                    "x": x_full,
                    "y": y,
                    "label": label,
                    "origin": "raw",
                    "mask": np.isfinite(x_full) & np.isfinite(y)
                })

        self.setWindowTitle(f"STS Viewer - {os.path.basename(self.filename)}")
        self.resize(900, 600)
        self.setMinimumSize(820, 400)

        # --- Central widget and layout ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ------------------------------ Tabs ------------------------------
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- Tab 1: Toolbar, Splitter (Left panel, Peak panel, Right panel) ---
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.setSpacing(6)

        # Toolbar
        self.toolbar = STSToolbar(parent=self)

        # Connect toolbar actions
        self.toolbar.action_save_png.triggered.connect(self.save_chart)
        self.toolbar.action_export_csv.triggered.connect(self.export_data)
        self.toolbar.action_norm_IU.triggered.connect(lambda: self.normalize("IU"))
        self.toolbar.action_norm_Imax.triggered.connect(lambda: self.normalize("Imax"))
        self.toolbar.action_savgol.triggered.connect(self.savgol_filter)
        self.toolbar.action_wiener.triggered.connect(self.wiener_filter)
        self.toolbar.btn_derivative.clicked.connect(self.calculate_derivative)
        self.toolbar.btn_subtract.clicked.connect(lambda: self.curve_arithmetic("subtract"))
        self.toolbar.btn_divide.clicked.connect(lambda: self.curve_arithmetic("divide"))
        self.toolbar.chk_swap_order.toggled.connect(lambda s: setattr(self, "reverse_order", s))
        self.toolbar.btn_fit.clicked.connect(self.fit_gauss)
        self.toolbar.btn_update.clicked.connect(
            lambda: self.plotter.plot_curves(indices=self.get_selected_indices(), average=False)
        )
        self.toolbar.btn_chk_all.clicked.connect(self.toggle_all_curves)
        self.toolbar.btn_hide.clicked.connect(self.hide_curves)
        self.toolbar.btn_average.clicked.connect(
            lambda: self.plotter.plot_curves(indices=self.get_selected_indices(), average=True)
        )
        self.toolbar.btn_reset.clicked.connect(self.reset_modifications)

        plot_layout.addWidget(self.toolbar)

        # ---------- Panels ----------
        # Splitter: Left panel + Peak panel + Right panel
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        Style.style_splitter(self.splitter)

        # --- Left panel: Plot ---
        fig = Figure()
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvas(fig)
        Style.style_canvas(self.canvas)
        self.splitter.addWidget(self.canvas)

        self.plotter = STSPlotter(self)

        # --- Peak panel (hidden initially) ---
        self.peak_panel = QWidget()
        Style.style_peak_panel(self.peak_panel)

        # Peak wrapper - add spacing around peak panel
        self.peak_wrapper = QWidget()
        p_wrapper_layout = QVBoxLayout(self.peak_wrapper)
        p_wrapper_layout.setContentsMargins(6, 0, 6, 0)
        p_wrapper_layout.setSpacing(8)
        p_wrapper_layout.addWidget(self.peak_panel)
        self.peak_wrapper.setMinimumWidth(180)
        self.peak_wrapper.setVisible(False)

        self.splitter.addWidget(self.peak_wrapper)

        # --- Right panel: Checklist buttons, Checklist title, Checklist, Manual input ---
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_panel.setObjectName("rightPanel")
        Style.style_right_panel(self.right_panel)

        # Checklist buttons
        btn_layout = QHBoxLayout()
        for btn in [self.toolbar.btn_update, self.toolbar.btn_chk_all,
                    self.toolbar.btn_hide, self.toolbar.btn_reset]:
            Style.style_checklist_icon(btn)
            btn_layout.addWidget(btn)
        btn_layout.addStretch()
        self.right_layout.addLayout(btn_layout)

        # Checklist title
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        select_label = QLabel("| SELECT CURVES:")
        Style.style_title(select_label)
        self.right_layout.addWidget(select_label)

        # Checklist
        self.curve_list = QListWidget()
        self.curve_list.itemClicked.connect(self.toggle_label_click)
        Style.style_curve_list(self.curve_list)
        scroll.setWidget(self.curve_list)
        self.right_layout.addWidget(scroll)

        # Manual input
        manual_layout = QHBoxLayout()
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("1, 2, 4-8")
        input_label = QLabel("INPUT:")
        Style.style_input_label(input_label)
        input_label.setToolTip(
            "<html>"
            "Enter original curves indexes manually. Index 0 stands for <i>All Curves</i>"
            "</html>"
        )
        validator = QRegularExpressionValidator(QRegularExpression(r"^[0-9,\-\s]*$"))
        self.manual_input.setValidator(validator)
        self.manual_input.returnPressed.connect(self.apply_manual_input)
        manual_layout.addWidget(input_label)
        manual_layout.addWidget(self.manual_input)

        self.right_layout.addLayout(manual_layout)

        # Right wrapper - add spacing around right panel
        self.right_wrapper = QWidget()
        r_wrapper_layout = QVBoxLayout(self.right_wrapper)
        r_wrapper_layout.setContentsMargins(6, 0, 0, 0)
        r_wrapper_layout.setSpacing(10)
        r_wrapper_layout.addWidget(self.right_panel)
        self.right_wrapper.setMinimumWidth(180)

        self.splitter.addWidget(self.right_wrapper)

        self.splitter.setStretchFactor(0, 2)  # left panel
        self.splitter.setStretchFactor(1, 1)  # peak panel
        self.splitter.setStretchFactor(2, 1)  # right panel

        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)

        plot_layout.addWidget(self.splitter)

        tabs.addTab(plot_tab, "📉 SPECTROSCOPY")

        # --- Tab 2: Search line, Metadata table ---
        self.metadata_tab = MetadataTab(channel_data=self.channel_data)
        self.metadata_tab.set_metadata(self.channel_data.get("attrs", {}))
        tabs.addTab(self.metadata_tab, "ℹ️ METADATA")
        tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)

        # Initial plotting & checklist update
        self.populate_curve_list(check_policy="checked")
        self.plotter.plot_curves()

    # ------------------------- Managing checklist functions -------------------------
    def populate_curve_list(self, check_policy="auto"):
        prev_checked = [
            self.curve_list.item(i).checkState() if self.curve_list.item(i) else None
            for i in range(self.curve_list.count())
        ]

        self.curve_list.blockSignals(True)
        self.curve_list.clear()

        plotted_all_idxs = set(self.plotter.plotted_curves)

        used_labels = {}
        for idx, entry in enumerate(self.all_curves):
            base = entry.get("label") or f"C{idx + 1}"
            used_labels[base] = used_labels.get(base, -1) + 1
            display = base if used_labels[base] == 0 else f"{base} ({used_labels[base]})"

            item = QListWidgetItem(display)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)

            if check_policy == "unchecked":
                item.setCheckState(Qt.CheckState.Unchecked)

            elif check_policy == "checked":
                item.setCheckState(Qt.CheckState.Checked)

            elif check_policy == "preserve":
                if idx < len(prev_checked) and prev_checked[idx] is not None:
                    item.setCheckState(prev_checked[idx])
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)

            else:  # "auto"
                if idx < len(prev_checked) and prev_checked[idx] is not None:
                    item.setCheckState(prev_checked[idx])
                else:
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if idx in plotted_all_idxs
                        else Qt.CheckState.Unchecked
                    )

            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.curve_list.addItem(item)

        self.curve_list.blockSignals(False)

    def get_selected_indices(self):
        if self.curve_list.count() == 0:
            return []

        return [
            i for i in range(self.curve_list.count())
            if self.curve_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def toggle_all_curves(self):
        if self.curve_list.count() == 0:
            return

        all_checked = all(
            self.curve_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.curve_list.count())
        )

        new_state = (
            Qt.CheckState.Unchecked if all_checked
            else Qt.CheckState.Checked
        )

        self.curve_list.blockSignals(True)
        for i in range(self.curve_list.count()):
            self.curve_list.item(i).setCheckState(new_state)
        self.curve_list.blockSignals(False)

    def toggle_label_click(self, item):
        pos = self.curve_list.viewport().mapFromGlobal(QCursor.pos())
        item_rect = self.curve_list.visualItemRect(item)

        option = QStyleOptionViewItem()
        option.initFrom(self.curve_list)
        option.rect = item_rect
        option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator

        check_rect = self.curve_list.style().subElementRect(
            QStyle.SubElement.SE_ItemViewItemCheckIndicator,
            option
        )

        if not check_rect.contains(pos):
            self.curve_list.blockSignals(True)
            try:
                item.setCheckState(
                    Qt.CheckState.Unchecked
                    if item.checkState() == Qt.CheckState.Checked
                    else Qt.CheckState.Checked
                )
            finally:
                self.curve_list.blockSignals(False)

    def apply_manual_input(self):
        text = self.manual_input.text().strip()

        # If empty: uncheck all curves
        if not text:
            for i in range(self.curve_list.count()):
                it = self.curve_list.item(i)
                it.setCheckState(Qt.CheckState.Unchecked)
            return

        # If user entered "0", select all curves
        if "0" in [t.strip() for t in text.replace("-", ",").split(",")]:
            for i in range(self.curve_list.count()):
                self.curve_list.item(i).setCheckState(Qt.CheckState.Checked)
            return

        positions = []

        def is_int(s: str) -> bool:
            s = s.strip()
            if not s:
                return False
            if s[0] in "+-" and len(s) > 1:
                return s[1:].isdigit()
            return s.isdigit()

        for part in text.split(","):
            part = part.strip()
            if not part:
                continue

            if "-" in part:
                tokens = [t.strip() for t in part.split("-")]
                if len(tokens) == 2 and is_int(tokens[0]) and is_int(tokens[1]):
                    start, end = int(tokens[0]), int(tokens[1])
                    step = 1 if end >= start else -1
                    for i in range(start, end + step, step):
                        positions.append(i - 1)
                continue

            if is_int(part):
                positions.append(int(part) - 1)

        # Deduplicate & validate
        valid = []
        seen = set()
        n = len(self.all_curves)
        for p in positions:
            if 0 <= p < n and p not in seen:
                seen.add(p)
                valid.append(p)

        # Clear all checkboxes first
        for i in range(self.curve_list.count()):
            self.curve_list.item(i).setCheckState(Qt.CheckState.Unchecked)

        # Apply new check states
        for p in valid:
            it = self.curve_list.item(p)
            if it:
                it.setCheckState(Qt.CheckState.Checked)

    # ------------------------- Toolbar operations -------------------------
    # --- Save / Export ---
    def save_chart(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", "", "PNG Files (*.png);;All Files (*)"
        )
        if file_path:
            self.canvas.figure.savefig(file_path)
            QMessageBox.information(self, "Saved", f"Chart saved to:\n{file_path}")

    def export_data(self):
        # Safety checks
        if not getattr(self, "all_curves", None):
            QMessageBox.warning(self, "No Data", "No curves available to export.")
            return

        # File dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Curves", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        # Remove existing file if present (replace)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot overwrite existing file:\n{e}")
                return

        # Metadata source
        attrs = self.channel_data.get("attrs", {})

        # X-axis from first curve
        x0 = np.array(self.all_curves[0]["x"])
        n_points = len(x0)

        # Cell values sanitization for CSV format
        def sanitize_metadata_value(value):
            s = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
            return f'{s}'

        # Collect all Y curves and labels
        y_arrays = []
        labels = []

        for entry in self.all_curves:
            y = np.array(entry.get("y", []))
            # Pad to match X-axis length
            if len(y) < n_points:
                y = np.pad(y, (0, n_points - len(y)), mode="constant", constant_values=np.nan)
            elif len(y) > n_points:
                y = y[:n_points]
            y_arrays.append(y)
            short = entry.get("label") or "Unnamed"
            labels.append(sanitize_metadata_value(short))
        try:
            with open(file_path, "w", newline='', encoding="utf-8", errors="replace") as f:
                writer = csv.writer(f)

                # Metadata rows
                metadata_keys = ["Attribute"]
                metadata_values = ["Value"]

                preferred_order = [
                    "filename", "RHK_Label", "long_name", "RHK_SessionText", "RHK_Bias",
                    "RHK_Ysize", "RHK_Xsize", "RHK_Xlabel",
                    "RHK_Ylabel", "RHK_Xscale", "RHK_Yscale",
                    "RHK_Xunits", "RHK_Yunits", "RHK_Zunits"
                ]

                # Add preferred keys first if exist
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

                # Prepare dynamic X-axis label
                xlabel = self.channel_data.get("attrs", {}).get("RHK_Xlabel", "Bias")
                xunits = self.channel_data.get("attrs", {}).get("RHK_Xunits", "V")

                # Header row
                header = ["Index", f"{xlabel} [{xunits}]"] + labels
                writer.writerow(header)

                # Data rows
                for i in range(n_points):
                    row = [i, x0[i]] + [y_arrays[j][i] for j in range(len(y_arrays))]
                    writer.writerow(row)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data:\n{e}")
            return

        QMessageBox.information(self, "Exported", f"Data saved to:\n{file_path}")

    # --- Normalization by I/U or by Imax ---
    def normalize(self, mode="IU"):
        indices = self.get_selected_indices()
        if not indices:
            QMessageBox.warning(self, "No Selection", "Please select at least one curve.")
            return

        # Validate current channel
        if not self.current_channel or not isinstance(self.current_channel, dict):
            QMessageBox.information(
                self, "Normalization Skipped",
                "No Current channel found. Normalization cannot be performed."
            )
            return

        current_data = self.current_channel.get("data")
        if current_data is None:
            QMessageBox.information(
                self, "Normalization Skipped",
                "No Current channel found. Normalization cannot be performed."
            )
            return

        # Get bias axis
        bias = STSOperations.get_bias_axis(self.channel_data)
        if bias.size == 0:
            QMessageBox.warning(self, "No Bias Data", "Bias axis could not be determined.")
            return

        # Process selected curves
        new_entries = []
        for idx in indices:
            if idx < 0 or idx >= len(self.all_curves):
                continue

            curve = self.all_curves[idx]
            parents = curve.get("parents") or [idx]
            if parents:
                parent_idx = parents[0]
            else:
                parent_idx = idx  # fallback if parents is empty
            if parent_idx < 0 or parent_idx >= current_data.shape[1]:
                continue

            current_column = current_data[:, parent_idx]
            entry = STSOperations.create_normalized_entry(curve, current_column, bias, mode)
            if entry is not None:
                new_entries.append(entry)

        if not new_entries:
            QMessageBox.information(self, "No Valid Curves",
                                    "No valid curves found for normalization.")
            return

        # Add new curves
        start_idx = len(self.all_curves)
        self.all_curves.extend(new_entries)
        new_indices = list(range(start_idx, start_idx + len(new_entries)))

        # Plot and update checklist
        self.plotter.plot_curves(indices=new_indices)
        self.populate_curve_list(check_policy="auto")

    # --- Filter curve with Savitzky-Golay or Wiener model ---
    def savgol_filter(self):
        indices = self.get_selected_indices()
        if not indices:
            QMessageBox.warning(self, "No Selection", "Please check at least one curve to filter.")
            return

        window_length, ok1 = QInputDialog.getInt(
            self, "Savitzky–Golay Filter",
            "Window length - odd number [3, 301]:", 21, 3, 301, 2
        )
        if not ok1 or window_length % 2 == 0:
            QMessageBox.warning(self, "Invalid Value", "Window length must be odd.")
            return

        polyorder, ok2 = QInputDialog.getInt(
            self, "Savitzky–Golay Filter",
            "Polynomial order - [1, 10]:", 3, 1, 10, 1
        )
        if not ok2 or polyorder >= window_length:
            QMessageBox.warning(self, "Invalid Value", "Polynomial order must be smaller than window length.")
            return

        new_entries = STSOperations.savgol_filter(self.all_curves, indices, window_length, polyorder)
        if not new_entries:
            QMessageBox.warning(self, "No Valid Curves", "No valid curves found for filtering.")
            return

        start_idx = len(self.all_curves)
        self.all_curves.extend(new_entries)
        new_indices = list(range(start_idx, len(self.all_curves)))

        self.plotter.plot_curves(indices=new_indices)
        self.populate_curve_list(check_policy="auto")

    def wiener_filter(self):
        indices = self.get_selected_indices()
        if not indices:
            QMessageBox.warning(self, "No Selection", "Please check at least one curve to filter.")
            return

        mysize, ok = QInputDialog.getInt(
            self, "Wiener Filter",
            "Neighborhood size:", 9, 1, 101, 1
        )
        if not ok:
            return


        new_entries = STSOperations.wiener_filter(self.all_curves, indices, mysize)
        if not new_entries:
            QMessageBox.warning(self, "No Valid Curves", "No valid curves found for filtering.")
            return

        start_idx = len(self.all_curves)
        self.all_curves.extend(new_entries)
        new_indices = list(range(start_idx, len(self.all_curves)))

        self.plotter.plot_curves(indices=new_indices)
        self.populate_curve_list(check_policy="auto")

    # --- Differentiate (math or lock-in) ---
    def calculate_derivative(self):
        indices = self.get_selected_indices()
        if not indices:
            QMessageBox.warning(self, "No Selection", "Please select at least one curve.")
            return

        amp, fmod, tau, ok = self.ask_lockin_parameters()
        if not ok:
            return

        use_lockin = amp is not None

        new_entries = STSOperations.compute_derivative(self.all_curves, indices, amp, fmod, tau, use_lockin)

        if not new_entries:
            QMessageBox.information(self, "No Result", "No valid curves for differentiation.")
            return

        start_idx = len(self.all_curves)
        self.all_curves.extend(new_entries)
        new_indices = list(range(start_idx, len(self.all_curves)))

        self.plotter.plot_curves(indices=new_indices)
        self.populate_curve_list(check_policy="auto")

    def ask_lockin_parameters(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Lock-in parameters")

        layout = QFormLayout(dlg)

        amp_field = QLineEdit()
        fmod_field = QLineEdit()
        tau_field = QLineEdit()

        layout.addRow("amp* [V]   =", amp_field)
        layout.addRow("f_mod [Hz] =", fmod_field)
        layout.addRow("τ [s]      =", tau_field)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(btns)

        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, None, None, False

        def parse(field):
            text = field.text().strip()
            return float(text) if text else None

        return parse(amp_field), parse(fmod_field), parse(tau_field), True

    # --- Subtract / Divide ---
    def curve_arithmetic(self, operation):
        indices = self.get_selected_indices()
        if len(indices) != 2:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly two curves.")
            return

        if not self.all_curves:
            QMessageBox.warning(self, "No Curves", "No curves available for operation.")
            return

        # Handle reverse order (switch)
        if getattr(self, "reverse_order", False):
            idxA, idxB = indices[1], indices[0]
        else:
            idxA, idxB = indices[0], indices[1]

        try:
            result_entry = STSOperations.curve_arithmetic(
                self.all_curves, idxA, idxB, operation
            )
        except ValueError as e:
            QMessageBox.warning(self, "Operation Error", str(e))
            return

        self.all_curves.append(result_entry)
        new_idx = len(self.all_curves) - 1

        self.plotter.plot_curves(indices=[idxA, idxB, new_idx])
        self.populate_curve_list(check_policy="auto")

    # --- Fit Gaussian ---
    def fit_gauss(self):
        selected = self.get_single_selected_curve()
        if selected is None:
            return

        _, _, _, base_index = selected

        # Create fitting object with selected curve index
        fitter = FitGauss(self, base_index)

        # Run full fitting workflow
        res = fitter.run()
        if res is None:
            return

    def add_fitted_curve_to_checklist(
            self, x, y, base_index, fit_result, fit_components,
            r2_local=None, r2_global=None):

        label = f"C{base_index + 1}-fitGauss"

        entry = {
            "x": np.asarray(x, float),
            "y": np.asarray(y, float),
            "label": label,
            "origin": "fit",
            "is_fit": True,
            "parents": [base_index],
            "fit_result": fit_result,
            "fit_components": fit_components,
            "r2_local": r2_local,
            "r2_global": r2_global,
        }

        # Handle duplicates
        existing = [
            c for c in self.all_curves
            if c.get("origin") == "fit" and c.get("parents", []) == [base_index]
        ]

        if existing:
            suffix = len(existing)
            entry["label"] = f"{label} ({suffix})"

        self.all_curves.append(entry)
        new_index = len(self.all_curves) - 1

        if hasattr(self, "populate_curve_list"):
            self.populate_curve_list()

        # Update curves checklist
        item = self.curve_list.item(new_index)
        if item:
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setCheckState(Qt.CheckState.Checked)
            item.setSelected(True)

        # Connect double-click
        if not getattr(self, "fit_doubleclick_connected", False):
            def handle_doubleclick(clicked_item):
                data = clicked_item.data(Qt.ItemDataRole.UserRole)
                if not data or not data.get("is_fit"):
                    return

                fitter = FitGauss(self, curve_index=data["parents"][0])
                fitter.show_fit_report(data)

            try:
                self.curve_list.itemDoubleClicked.connect(handle_doubleclick)
                self.fit_doubleclick_connected = True
            except Exception as e:
                logger.warning(f"Failed to connect double-click: {e}")

        return new_index

    # --- Reset all modifications ---
    def reset_modifications(self):
        if self.original_data is None:
            QMessageBox.information(self, "No Data", "No original data available to reset.")
            return

        # Restore original data
        try:
            self.modified_data = np.asarray(self.original_data, dtype=float)
        except Exception as e:
            QMessageBox.critical(self, "Reset Error", f"Data conversion failed:\n{e}")
            return

        # Rebuild curves
        attrs = self.channel_data.get("attrs", {})
        n_points, n_curves = self.modified_data.shape
        x_offset = float(attrs.get("RHK_Bias", 0) or 0)
        x_scale = float(attrs.get("RHK_Xscale", 1) or 1)
        x_full = np.linspace(x_offset, x_offset + x_scale * (n_points - 1), n_points, dtype=float)

        self.all_curves = []
        for i in range(n_curves):
            y = np.asarray(self.modified_data[:, i], dtype=float)
            y = np.nan_to_num(y, nan=np.nan, posinf=np.nan, neginf=np.nan)

            entry = STSOperations.create_new_entry(
                curve={"x": x_full, "y": y, "label": f"C{i + 1}", "parents": [i]},
                y_new=y,
                label_prefix=None,
                origin="raw"
            )
            # For raw data, parents should point to itself
            entry["parents"] = [i]
            self.all_curves.append(entry)

        # Clear plot and update the checklist
        self.curves = []
        self.populate_curve_list(check_policy="unchecked")
        self.plotter.clear_plot()

        # Hide peak panel, clear context
        if hasattr(self, "peak_wrapper") and self.peak_wrapper is not None:
            self.peak_wrapper.setVisible(False)
        if hasattr(self, "last_run_context"):
            self.last_run_context = None

        # Reset switch checkbox
        if hasattr(self.toolbar, "chk_swap_order"):
            self.toolbar.chk_swap_order.setChecked(False)

        # Clear manual input field
        if hasattr(self, "manual_input"):
            self.manual_input.clear()

    # --- Hide curves: clear plot and uncheck all curves ---
    def hide_curves(self):
        # Uncheck checklist items
        for i in range(self.curve_list.count()):
            item = self.curve_list.item(i)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)

        # Clear plotted curves
        self.plotter.clear_plot()

    #  --- Helpers tools ---
    def get_single_selected_curve(self):
        indices = self.get_selected_indices()
        if len(indices) != 1:
            QMessageBox.warning(self, "Invalid Selection",
                                "Please select exactly one curve.")
            return None

        idx = indices[0]

        # Validation
        if not hasattr(self, "all_curves") or idx < 0 or idx >= len(self.all_curves):
            QMessageBox.critical(
                self, "Index Error",
                f"Selected curve #{idx + 1} does NOT exist.\n"
                f"Available curves: 1–{len(self.all_curves)}."
            )
            return None

        entry = self.all_curves[idx]
        x = np.asarray(entry["x"], float)
        y = np.asarray(entry["y"], float)

        # Find label
        label = entry.get("label") or f"C{idx + 1}"

        return x, y, label, idx
