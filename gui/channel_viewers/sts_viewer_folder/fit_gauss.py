from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QMessageBox, QApplication,
    QDialog, QToolButton, QScrollArea, QLabel, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
import numpy as np
from scipy.signal import find_peaks
from lmfit.models import GaussianModel
import logging

from gui.helpers.styles import Style
from gui.channel_viewers.sts_viewer_folder.sts_processing import STSOperations


logger = logging.getLogger(__name__)

class FitGauss:
    def __init__(self, viewer, curve_index=None):
        self.viewer = viewer
        self.curve_index = curve_index
        self.ax = viewer.ax
        self.canvas = viewer.canvas

    # ------------------------- Main workflow entry -------------------------
    def run(self):
        selected = self.viewer.get_single_selected_curve()
        if selected is None:
            return None

        x, y, label, idx = selected

        # Use the last curve version
        entry = self.viewer.all_curves[idx]
        bias = STSOperations.get_bias_axis(self.viewer.channel_data)
        x, y, label = self.prepare_curve(entry, bias_axis=bias, idx=idx)

        # Ensure curve is plotted
        plotted_indices = [it_idx for _, _, it_idx in getattr(self.viewer, "curves", [])]
        if idx not in plotted_indices:
            try:
                self.viewer.plotter.plot_curves(indices=[idx], average=False)
            except Exception:
                # fallback: plot raw x,y
                try:
                    self.viewer.ax.plot(x, y, label=label)
                    self.viewer.canvas.draw_idle()
                except Exception as e:
                    logger.warning("Unable to plot raw data:", e)

        # Auto-detect peaks
        peaks = self.auto_detect_peaks(x, y)
        if len(peaks) == 0:
            self.viewer.show_error("No peaks detected automatically.")
            return None

        # Draw temporary preview
        self.draw_peak_preview(peaks)

        # Store curve context for panel callbacks
        self.last_run_context = {
            "x": x,
            "y": y,
            "label": label,
            "idx": idx
        }

        # Show peak panel
        self.show_peak_panel(peaks, x=x, y=y, label=label, idx=idx)

    def prepare_curve(self, entry, bias_axis=None, idx=None):
        x = np.asarray(entry["x"], dtype=float)
        y = np.asarray(entry["y"], dtype=float)

        # Flatten arrays
        if x.shape != y.shape or x.ndim > 1:
            x = np.concatenate(x).ravel()
            y = np.concatenate(y).ravel()
        else:
            x = x.ravel()
            y = y.ravel()

        # Override x with bias axis if provided
        if bias_axis is not None and len(bias_axis) == len(x):
            x = np.asarray(bias_axis, dtype=float)

        # Generate label
        label = entry.get("short") or entry.get("label") or (f"C{idx + 1}" if idx is not None else "Unknown")

        return x, y, label

    def on_peak_panel_ok(self, peaks, x, y, label, idx):
        # Perform fit
        fit_x, fit_y, fit_result, components, r2_global, r2_local = self.perform_fit(x, y, peaks)

        # Prepare label for legend
        short_label = f"{label}-FitGauss"

        # Plot fitted curve + components
        self.viewer.ax.clear()
        self.viewer.plotter.plot_curves(indices=[idx], average=False)
        self.viewer.ax.plot(fit_x, fit_y, '--', lw=2.0, label=short_label)
        for i, gm in enumerate(components):
            y_comp = gm.eval(fit_result.params, x=fit_x) + np.min(y)
            self.viewer.ax.plot(fit_x, y_comp, linestyle=':', alpha=0.6, color='gray')
        self.viewer.ax.legend()
        self.viewer.canvas.draw_idle()

        # Add fitted curve to checklist
        new_index = self.viewer.add_fitted_curve_to_checklist(
            x=fit_x,
            y=fit_y,
            base_index=idx,
            fit_result=fit_result,
            fit_components=components,
            r2_local=r2_local,
            r2_global=r2_global
        )
        fit_entry = self.viewer.all_curves[new_index]

        # Show fit report
        self.show_fit_report(fit_entry)
        return fit_entry

    def auto_detect_peaks(self, x, y, max_peaks=20):
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        # Baseline shift
        y_shift = y - np.min(y)

        # Smoothing (simple moving average)
        window = max(1, int(len(y) * 0.01))  # ~1% of total points
        if window > 1:
            y_smooth = np.convolve(y_shift, np.ones(window) / window, mode="same")
        else:
            y_smooth = y_shift

        # Find peaks
        peaks, _ = find_peaks(
            y_smooth,
            height=np.max(y_smooth) * 0.05,  # only peaks >5% of max
            distance=int(len(x) * 0.05),  # min distance between peaks (~5% of axis)
            prominence=np.max(y_smooth) * 0.05
        )

        # Limit number of peaks
        peak_xs = np.sort(x[peaks])[:max_peaks]

        self.auto_peaks = peak_xs
        return peak_xs

    def show_peak_panel(self, peaks, x=None, y=None, label=None, idx=None):
        peaks = np.asarray(peaks, float)

        self.edit_x = np.array(x, float) if x is not None else None
        self.edit_y = np.array(y, float) if y is not None else None
        self.edit_label = label
        self.edit_idx = idx

        self.viewer.peak_wrapper.setVisible(True)
        panel = self.viewer.peak_panel
        layout = panel.layout()

        if layout is None:
            layout = self.peak_panel_layout(panel)
        else:
            self.clear_layout(layout)

        # Header
        header = QLabel("| DETECTED PEAKS")
        Style.style_title(header)
        text = QLabel(
            f"<b>{len(peaks)}</b> peak(s) detected.<br><br>"
            "- Select <i>Edit</i> to choose peaks manually<br>"
            "- Select <i>OK</i> to continue."
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)
        layout.addWidget(text)

        coords = Style.peak_coords_detected(peaks)
        layout.addWidget(coords, stretch=1)

        # Preview lines
        self.draw_peak_preview(peaks)

        def edit_cb():
            self.clear_peak_preview()
            self.manual_select_peaks()

        # Buttons
        btn_layout = QHBoxLayout()
        layout.addLayout(btn_layout)

        self.btn_edit = self.add_button(
            "✏️ Edit",
            "Manually select peak positions",
            btn_layout,
            edit_cb
        )

        self.btn_ok = self.add_button(
            "✅ OK",
            "Confirm detected peaks and proceed to fitting",
            btn_layout,
            lambda: self.on_peak_panel_ok(peaks, self.edit_x, self.edit_y,
                                          self.edit_label, self.edit_idx)
        )

        panel.setVisible(True)
        panel.update()

    def manual_select_peaks(self, max_peaks=20):
        panel = self.viewer.peak_panel
        layout = panel.layout()
        self.clear_layout(layout)

        # Header / instructions
        header = QLabel("| SELECT PEAKS")
        Style.style_title(header)
        text = QLabel(
            f"- Left click = add peak,\n"
            f"- Right click = undo last.\n\n"
            f"Limit: 1-{max_peaks} peaks"
        )
        text.setWordWrap(True)
        layout.addWidget(header)
        layout.addWidget(text)

        # Scrollable fields container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(Style.SCROLLBAR_VERTICAL)

        field_container = QWidget()
        field_layout = QVBoxLayout(field_container)
        field_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(2)

        scroll.setWidget(field_container)
        layout.addWidget(scroll, stretch=1)

        # --- Track state ---
        selected_peaks = []
        self.manual_preview_lines = []

        # --- Fields update ---
        def update_fields():
            # Clear old widgets
            while field_layout.count():
                item = field_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    while item.layout().count():
                        sub = item.layout().takeAt(0)
                        if sub.widget():
                            sub.widget().deleteLater()

            for i, val in enumerate(selected_peaks):
                row, le = Style.peak_row_manual(i + 1, val, is_first=(i == 0))
                field_layout.addLayout(row)

                def handler(text, idx=i):
                    try:
                        selected_peaks[idx] = float(text)
                        redraw_preview()
                    except ValueError:
                        pass

                le.textChanged.connect(handler)

        # --- Preview redraw ---
        def redraw_preview():
            for ln in self.manual_preview_lines:
                ln.remove()
            self.manual_preview_lines.clear()

            for val in selected_peaks:
                ln = self.ax.axvline(val, linestyle="--", color="blue", alpha=0.7)
                self.manual_preview_lines.append(ln)

            self.canvas.draw_idle()

        # --- Mouse click handler ---
        def on_click(event):
            if event.inaxes != self.ax:
                return

            # Left click - add
            if event.button == 1 and event.xdata is not None:
                if len(selected_peaks) >= max_peaks:
                    return
                selected_peaks.append(float(event.xdata))
                update_fields()
                redraw_preview()

            # Right click - undo
            elif event.button == 3 and selected_peaks:
                selected_peaks.pop()
                update_fields()
                redraw_preview()

        cid = self.canvas.figure.canvas.mpl_connect("button_press_event", on_click)
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

        # --- Cleanup function restores default cursor ---
        def cleanup():
            self.canvas.figure.canvas.mpl_disconnect(cid)
            for ln in self.manual_preview_lines:
                ln.remove()
            self.manual_preview_lines.clear()
            self.canvas.draw_idle()
            self.clear_layout(panel.layout())

            # Restore default arrow cursor after selection ends
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        def cancel():
            cleanup()  # remove temporary manual peaks

            # Restore auto-detected peaks
            auto_peaks = getattr(self, "auto_peaks", [])
            if auto_peaks is None or len(auto_peaks) == 0:
                return  # nothing to restore

            # Restore internal curve context
            x_data = self.edit_x
            y_data = self.edit_y
            label = self.edit_label
            idx = self.edit_idx

            self.show_peak_panel(auto_peaks, x=x_data, y=y_data, label=label, idx=idx)
            self.draw_peak_preview(auto_peaks)
            self.panel_result = {"action": "cancel", "peaks": auto_peaks}

        def done():
            if not selected_peaks:
                cancel()  # use stored auto_peaks
                return

            peaks = np.sort(np.array(selected_peaks, float))
            cleanup()

            # Retrieve curve data
            x_data = self.edit_x
            y_data = self.edit_y
            label = self.edit_label
            idx = self.edit_idx

            fit_entry = self.on_peak_panel_ok(peaks, x_data, y_data, label, idx)
            if fit_entry is not None:
                self.show_fit_report(fit_entry)

        # Buttons
        btn_layout = QHBoxLayout()
        layout.addLayout(btn_layout)

        self.btn_cancel = self.add_button(
            "❌ Cancel",
            "Cancel selection and return to detected peaks",
            btn_layout,
            cancel
        )

        self.btn_done = self.add_button(
            "✅ Done",
            "Confirm selection and proceed to fitting",
            btn_layout,
            done
        )

    def perform_fit(self, x, y, peaks):
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        # --- Baseline shift ---
        baseline = float(np.min(y))
        y_shift = y - baseline

        gaussians = []
        model = None
        params = None
        x_range = float(np.max(x) - np.min(x)) if len(x) > 1 else 1.0
        masks = []

        # --- Build model and parameters ---
        for i, center in enumerate(peaks):
            prefix = f"g{i}_"
            gm = GaussianModel(prefix=prefix)
            gaussians.append(gm)
            model = gm if model is None else model + gm

            if params is None:
                params = gm.make_params()
            else:
                params.update(gm.make_params())

            # Local window for initial estimates
            local_half = max(x_range * 0.03, (x[1] - x[0]) * 5)
            mask = (x >= center - local_half) & (x <= center + local_half)
            masks.append(mask)

            xloc = x[mask]
            yloc = y_shift[mask]

            local_max_idx = int(np.argmax(yloc))
            amp_guess = float(yloc[local_max_idx])
            sigma_guess = max(np.ptp(xloc) / 2.355, (x[1] - x[0]) * 0.5)

            params[prefix + "center"].set(value=center, min=center - local_half, max=center + local_half)
            params[prefix + "amplitude"].set(value=amp_guess, min=0)
            params[prefix + "sigma"].set(value=sigma_guess, min=sigma_guess / 8.0, max=sigma_guess * 8.0)

        # --- Combined mask for local R2 ---
        combined_mask = np.zeros_like(x, dtype=bool)
        for m in masks:
            combined_mask |= m
        x_mask = x[combined_mask]
        y_mask = y_shift[combined_mask]

        # --- Fit ---
        try:
            result = model.fit(y_mask, params, x=x_mask)
        except Exception as e:
            QMessageBox.critical(self.viewer, "Fit Error", f"Gaussian fit failed:\n{e}")
            return x, y, None, [], None, None  # return r2_local

        # --- Full fitted curve ---
        y_fit_shifted = model.eval(result.params, x=x)
        y_fit_full = y_fit_shifted + baseline

        # --- Global R2 ---
        residuals_global = y - y_fit_full
        ss_res_global = np.sum(residuals_global ** 2)
        ss_tot_global = np.sum((y - np.mean(y)) ** 2)
        r2_global = 1 - ss_res_global / ss_tot_global if ss_tot_global != 0 else None

        # --- Local R2 ---
        x_local = x[combined_mask]
        y_local = y[combined_mask]
        y_fit_local = y_fit_full[combined_mask]

        if len(x_local) > 5:
            residuals_local = y_local - y_fit_local
            ss_res_local = np.sum(residuals_local ** 2)
            ss_tot_local = np.sum((y_local - np.mean(y_local)) ** 2)
            r2_local = 1 - ss_res_local / ss_tot_local if ss_tot_local != 0 else None
        else:
            r2_local = None

        return x, y_fit_full, result, gaussians, r2_global, r2_local

    def show_fit_report(self, entry):
        viewer = self.viewer

        if entry is None or entry.get("origin") != "fit":
            viewer.show_error("Invalid entry for fit report.")
            return

        fit_result = entry.get("fit_result")
        components = entry.get("fit_components")
        r2_global = entry.get("r2_global", None)
        r2_local = entry.get("r2_local", None)

        if fit_result is None or components is None:
            viewer.show_error("Fit metadata is missing.")
            return

        # Build text report
        info_lines = [
            f"R\u00B2 global = {r2_global:.4f}" if r2_global is not None else "R\u00B2 global = N/A",
            f"R\u00B2 local  = {r2_local:.4f}\n" if r2_local is not None else "R\u00B2 local  = N/A\n",
        ]

        for i, gm in enumerate(components):
            prefix = f"g{i}_"
            try:
                U = float(fit_result.params[prefix + "center"].value)
                amp = float(fit_result.params[prefix + "amplitude"].value)
                sig = float(fit_result.params[prefix + "sigma"].value)
                info_lines.append(
                    f"Peak {i + 1}:\n"
                    f"  U = {U:.4f}\n"
                    f"  A = {amp:.4e}\n"
                    f'  σ = {sig:.4e}\n'
                )
            except Exception:
                logger.warning("Failed to extract parameters for peak %d in fit report.", i)

        info_text = "\n".join(info_lines)
        panel = viewer.peak_panel

        # Keep existing layout or create one if missing
        panel_layout = panel.layout()
        if panel_layout is None:
            panel_layout = self.peak_panel_layout(panel)
        else:
            self.clear_layout(panel.layout())

        # Populate layout with fit report
        header = QLabel("| FIT RESULTS")
        Style.style_title(header)
        panel_layout.addWidget(header)

        self.btn_fer = self.add_button(
            "⚛️ FER peak position",
            "Show Field Emission Resonance peak positions.",
            panel_layout,
            lambda: self.open_fer_peak_window(fit_result, components)
        )

        # --- Scrollable fit report ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(Style.SCROLLBAR_VERTICAL)

        # Content widget inside scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Fit report label
        report_label = QLabel(info_text)
        report_label.setWordWrap(True)
        report_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        report_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        report_label.setStyleSheet("padding-top: 4px; padding-left: 6px;")

        scroll_layout.addWidget(report_label)
        scroll_area.setWidget(scroll_content)
        panel_layout.addWidget(scroll_area)

        btn_layout = QHBoxLayout()
        panel_layout.addLayout(btn_layout)

        self.btn_copy = self.add_button(
            "📄 Copy",
            "Copy peak report data to clipboard",
            btn_layout,
            lambda: QApplication.clipboard().setText(info_text)
        )

        self.btn_close = self.add_button(
            "❌ Close",
            "Close the panel",
            btn_layout,
            lambda: self.viewer.peak_wrapper.setVisible(False)
        )

        # Ensure wrapper is visible
        if hasattr(self.viewer, "peak_wrapper") and self.viewer.peak_wrapper is not None:
            self.viewer.peak_wrapper.setVisible(True)

        # Show panel
        panel.setVisible(True)
        panel.update()

    def open_fer_peak_window(self, fit_result, gaussians):
        # Extract peak positions from fit
        peak_positions = []
        for i in range(len(gaussians)):
            prefix = f"g{i}_"
            try:
                U = float(fit_result.params[prefix + "center"].value)
                peak_positions.append(U)
            except Exception as e:
                logger.warning("Unable to update FER peak position:", e)
                continue

        n_peaks = len(peak_positions)
        if n_peaks == 0:
            QMessageBox.information(self.viewer, "FER Peaks", "No peaks found in fit.")
            return

        # X-axis: (n - 1/4)^(2/3)
        x = np.array([(n - 0.25) ** (2 / 3) for n in range(1, n_peaks + 1)])
        y = np.array(peak_positions)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept

        # Compute R2
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        # Create dialog
        fer_dlg = QDialog(self.viewer)
        fer_dlg.setWindowTitle("FER Peak Positions")
        layout = QVBoxLayout(fer_dlg)

        # Matplotlib figure
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='blue', s=50)

        x_min = -0.1
        y_min = slope * x_min + intercept
        ax.set_xlim(left=x_min)
        ax.set_ylim(bottom=y_min)

        ax.axhline(0, color='black', linewidth=1)  # Bold x-axis
        ax.axvline(0, color='black', linewidth=1)  # Bold y-axis

        x_line = np.linspace(x_min, max(x), 500)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='grey', linestyle='--',
                label=(f'Linear fit: y = {slope:.4f}x + {intercept:.4f}\n'
                       f'$R^2$ = {r2:.4f}\n'
                       f'Work function $\\phi$ = {intercept:.4f} eV'))
        ax.set_xlabel(r'$(n - 1/4)^{2/3}$', fontsize=12)
        ax.set_ylabel("Bias [V]", fontsize=12)
        ax.set_title("FER Peak Positions", fontsize=14)
        ax.minorticks_on()
        ax.grid(True, which='major', linewidth=0.6)
        ax.grid(True, which='minor', linewidth=0.3, alpha=0.4)
        ax.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        fer_dlg.resize(700, 500)
        fer_dlg.show()

    # ------------------------- Helper methods -------------------------
    def add_button(self, text, tooltip, layout, callback):
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        Style.style_buttons([btn])
        layout.addWidget(btn)
        btn.clicked.connect(callback)
        return btn

    def draw_peak_preview(self, peaks):
        # Remove old lines if exist
        if hasattr(self, "temp_lines"):
            for ln in self.temp_lines:
                try:
                    ln.remove()
                except Exception as e:
                    logger.warning("Unable to remove temporary peak preview line:", e)
            self.temp_lines.clear()

        self.temp_lines = []

        # Draw new lines
        if peaks is None or len(peaks) == 0:
            return

        for px in peaks:
            ln = self.viewer.ax.axvline(px, color="blue", linestyle="--", linewidth=1.5, alpha=0.6)
            self.temp_lines.append(ln)

        try:
            self.viewer.canvas.draw_idle()
        except Exception:
            self.viewer.canvas.draw()

    def clear_peak_preview(self):
        if hasattr(self, "temp_lines") and self.temp_lines:
            for ln in self.temp_lines:
                try:
                    ln.remove()
                except Exception as e:
                    logger.warning("Unable to remove temporary peak preview lines:", e)
            self.temp_lines.clear()
            try:
                self.viewer.canvas.draw_idle()
            except Exception:
                self.viewer.canvas.draw()

    def peak_panel_layout(self, widget):
        layout = widget.layout()
        if layout is None:
            layout = QVBoxLayout(widget)
            widget.setLayout(layout)
        else:
            self.clear_layout(layout)

        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        return layout

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                self.clear_layout(item.layout())
