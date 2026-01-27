from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt
from scipy.signal import savgol_filter, wiener
import numpy as np
import logging

logger = logging.getLogger(__name__)

class STSPlotter:
    def __init__(self, viewer):
        self.viewer = viewer
        self.ax = viewer.ax
        self.canvas = viewer.canvas
        self.plotted_curves = []

    def clear_plot(self):
        self.ax.clear()
        self.plotted_curves = []
        self.canvas.draw()

    def plot_curves(self, indices=None, average=False):
        self.ax.clear()
        self.plotted_curves = []

        all_curves = getattr(self.viewer, "all_curves", None)
        if all_curves is None or not all_curves:
            self.canvas.draw()
            return

        if indices is None:
            indices = list(range(len(all_curves)))
        elif indices == []:
            self.canvas.draw()
            return

        # --- Average mode ---
        if average:
            if len(indices) < 2:
                QMessageBox.warning(
                    self.viewer, "Invalid Selection",
                    "Please select two or more curves."
                )
                return

            xs, ys = [], []
            for i in indices:
                if 0 <= i < len(all_curves):
                    entry = all_curves[i]
                    xs.append(np.asarray(entry["x"], float))
                    ys.append(np.asarray(entry["y"], float))

            if not ys:
                return

            # Align X axes
            all_equal = all(np.array_equal(xs[0], x) for x in xs)
            if all_equal:
                common_x = xs[0]
                interp_ys = ys
            else:
                min_len = min(len(x) for x in xs)
                min_x = max(x[0] for x in xs)
                max_x = min(x[-1] for x in xs)
                common_x = np.linspace(min_x, max_x, min_len)
                interp_ys = [np.interp(common_x, x, y) for x, y in zip(xs, ys)]

            stack = np.vstack([np.asarray(y, float) for y in interp_ys])
            avg_y = np.nanmean(stack, axis=0)

            parent_labels = [all_curves[i].get("label", f"C{i + 1}") for i in indices]
            avg_label = f"(avg:{','.join(parent_labels)})"

            new_entry = STSOperations.create_new_entry(
                curve=all_curves[indices[0]],
                y_new=avg_y,
                label_prefix=None,
                origin="average"
            )
            new_entry["label"] = avg_label

            new_entry["x"] = common_x
            new_entry["mask"] = np.isfinite(common_x) & np.isfinite(avg_y)
            new_entry["parents"] = list(indices)

            # Persist average curve in viewer
            all_curves.append(new_entry)
            last_idx = len(all_curves) - 1

            # Plot only new average curve
            self.plot_curves(indices=[last_idx])
            self.viewer.populate_curve_list()

            # Auto-check new curve
            item = self.viewer.curve_list.item(last_idx)
            if item:
                item.setCheckState(Qt.CheckState.Checked)

            return

        # --- Normal plotting ---
        for i in indices:
            if 0 <= i < len(all_curves):
                curve_entry = all_curves[i]
                mask = curve_entry.get(
                    "mask", np.isfinite(curve_entry["x"]) & np.isfinite(curve_entry["y"])
                )
                self.ax.plot(curve_entry["x"][mask], curve_entry["y"][mask], label=curve_entry["label"])
        self.plotted_curves = indices.copy()

        # --- Axis labels ---
        attrs = self.viewer.channel_data.get("attrs", {}) or {}
        title = self.viewer.channel_data.get("title", "STS Data")
        x_label = attrs.get("RHK_Xlabel", "Bias")
        x_unit = attrs.get("RHK_Xunits", "V")

        if title.lower() == "current":
            y_label = "I"
            y_unit = attrs.get("RHK_Zunits", "A")
        elif title.lower() in ["di/du", "didu"]:
            y_label = "dI/dU"
            y_unit = "a.u."
        else:
            y_label = "Signal"
            y_unit = ""

        self.ax.set_xlabel(f"{x_label} [{x_unit}]")
        self.ax.set_ylabel(f"{y_label} [{y_unit}]")
        self.ax.set_title(title)

        if self.ax.lines:
            self.ax.legend()

        self.canvas.draw()


class STSOperations:
    @staticmethod
    def create_new_entry(curve, y_new, label_prefix=None, origin="custom"):
        x = np.asarray(curve["x"], dtype=float)
        y_new = np.asarray(y_new, dtype=float)

        if label_prefix is not None:
            label = f"({curve.get('label', 'C?')} [{label_prefix}])"
        else:
            label = curve.get("label", "C?")

        entry = {
            "x": x,
            "y": y_new,
            "label": label,
            "origin": origin,
            "parents": [],
            "mask": np.isfinite(x) & np.isfinite(y_new)
        }
        return entry

    @staticmethod
    def create_normalized_entry(curve, current_column, bias, mode="IU"):
        y = np.asarray(curve["y"], dtype=float)
        current = np.asarray(current_column, dtype=float)
        bias = np.asarray(bias, dtype=float)

        # Avoid division by zero
        bias_safe = np.where(bias == 0, 1e-16, bias)
        I_safe = np.where(current == 0, 1e-16, current)

        # Perform normalization
        if mode == "IU":
            y_norm = y * bias_safe / I_safe
            label_prefix = "nIU"
        elif mode == "Imax":
            i_max = np.max(np.abs(current))
            if i_max == 0:
                return None
            y_norm = y / i_max
            label_prefix = "nI"
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        # Build entry
        entry = STSOperations.create_new_entry(
            curve,
            y_new=y_norm,
            label_prefix=label_prefix,
            origin=f"normalize_{mode}"
        )
        entry["parents"] = curve.get("parents", [])
        return entry

    @staticmethod
    def savgol_filter(all_curves, indices, window_length, polyorder):
        return STSOperations.apply_filter_to_curves(
            all_curves,
            indices,
            filter_func=savgol_filter,
            label_suffix="SG",
            origin="savgol",
            window_length=window_length,
            polyorder=polyorder,
            mode="interp"
        )

    @staticmethod
    def wiener_filter(all_curves, indices, mysize):
        return STSOperations.apply_filter_to_curves(
            all_curves,
            indices,
            filter_func=wiener,
            label_suffix="W",
            origin="wiener",
            mysize=mysize
        )

    @staticmethod
    def apply_filter_to_curves(all_curves, indices, filter_func, label_suffix, origin, *args, **kwargs):
        new_entries = []

        for idx in indices:
            if idx < 0 or idx >= len(all_curves):
                continue

            curve = all_curves[idx]
            y = np.asarray(curve.get("y", []), dtype=float)
            if y.size == 0:
                continue

            try:
                y_filt = filter_func(y, *args, **kwargs)
            except Exception as e:
                logger.warning("Cannot perform filtering operation:", e)
                continue

            entry = STSOperations.create_new_entry(
                curve=curve,
                y_new=y_filt,
                label_prefix=label_suffix,
                origin=origin
            )
            entry["parents"] = [idx]
            new_entries.append(entry)
        return new_entries

    @staticmethod
    def compute_derivative(all_curves, indices, amp=None, fmod=None, tau=None, use_lockin=False):
        new_entries = []

        for idx in indices:
            if idx < 0 or idx >= len(all_curves):
                continue

            curve = all_curves[idx]
            x = np.asarray(curve.get("x", []), dtype=float)
            y = np.asarray(curve.get("y", []), dtype=float)

            if len(x) < 3:
                continue  # too short for differentiation

            # Compute derivative
            dy_dx = STSOperations.lockin_derivative(x, y, amp, fmod, tau) if use_lockin else np.gradient(y, x)

            # Build a unique label
            parent_label = curve.get("label") or f"C{idx + 1}"
            base_label = f"(d[{parent_label}]/dU)"
            existing_labels = {c.get("label") for c in all_curves}
            deriv_label = base_label
            counter = 1
            while deriv_label in existing_labels:
                deriv_label = f"{base_label} ({counter})"
                counter += 1

            entry = STSOperations.create_new_entry(
                curve,
                y_new=dy_dx,
                label_prefix=None,
                origin="derivative"
            )
            entry["label"] = deriv_label
            new_entries.append(entry)

        return new_entries

    @staticmethod
    def lockin_derivative(x, y, amp, fmod=None, tau=None):
        # Safety fallback
        if amp is None or amp <= 0:
            return np.gradient(y, x)

        # Check amplitude comparing to bias range and sampling step
        dx_min = np.min(np.diff(x))
        if amp > 0.5 * (x[-1] - x[0]):
            logger.warning("Amplitude larger than bias range. Derivative may be inaccurate.")
        elif amp > dx_min:
            logger.info("Amplitude larger than sampling step – derivative will be smoothed.")

        # Lock-in simulation
        Nph = 120
        phases = np.linspace(0, 2 * np.pi, Nph)
        y_out = np.zeros_like(y)
        xmin, xmax = x.min(), x.max()

        for phi in phases:
            shifted_x = x + amp * np.sin(phi)
            shifted_x = np.clip(shifted_x, xmin, xmax)
            y_shift = np.interp(shifted_x, x, y)
            y_out += y_shift * np.sin(phi)

        y_out /= Nph
        y_out /= amp

        # Lock-in low-pass filter
        if tau is not None and fmod is not None:
            alpha = np.exp(-1.0 / (2 * np.pi * fmod * tau))
            y_lp = np.zeros_like(y_out)
            y_lp[0] = y_out[0]
            for i in range(1, len(y_out)):
                y_lp[i] = alpha * y_lp[i - 1] + (1 - alpha) * y_out[i]
            return y_lp

        return y_out

    @staticmethod
    def curve_arithmetic(all_curves, idxA, idxB, operation):
        try:
            curve_a = all_curves[idxA]
            curve_b = all_curves[idxB]
        except Exception:
            raise ValueError("Curve data not found for selected indices.")

        a_x = np.asarray(curve_a["x"], dtype=float)
        a_y = np.asarray(curve_a["y"], dtype=float)
        b_x = np.asarray(curve_b["x"], dtype=float)
        b_y = np.asarray(curve_b["y"], dtype=float)

        # Ensure X-axis compatibility
        if not np.array_equal(a_x, b_x):
            xmin = max(a_x.min(), b_x.min())
            xmax = min(a_x.max(), b_x.max())
            if xmin >= xmax:
                raise ValueError("Curves have incompatible X axes.")

            n = min(len(a_x), len(b_x))
            x = np.linspace(xmin, xmax, n)
            a_y = np.interp(x, a_x, a_y)
            b_y = np.interp(x, b_x, b_y)
        else:
            x = a_x

        a_label = curve_a.get("label", f"C{idxA + 1}")
        b_label = curve_b.get("label", f"C{idxB + 1}")

        # Perform operation
        if operation == "subtract":
            y = a_y - b_y
            label = f"({a_label}-{b_label})"
        elif operation == "divide":
            b_safe = np.where(b_y == 0, np.nan, b_y)
            y = a_y / b_safe
            label = f"({a_label}/{b_label})"
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Build final entry
        entry = STSOperations.create_new_entry(
            curve={"x": x, **curve_a},
            y_new=y,
            label_prefix=None,
            origin=operation
        )
        entry["label"] = label
        entry["parents"] = [idxA, idxB]

        return entry

    @staticmethod
    def get_bias_axis(channel_data):
        attrs = channel_data.get("attrs", {}) or {}
        start = float(attrs.get("RHK_Bias", 0.0) or 0.0)
        scale = float(attrs.get("RHK_Xscale", 1.0) or 1.0)

        data = channel_data.get("data")
        if data is None:
            return np.array([], dtype=float)

        n_points = int(data.shape[0])
        return start + np.arange(n_points, dtype=float) * scale
