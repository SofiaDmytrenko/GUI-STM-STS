import os
import csv
import spym
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MeasurementData:
    def __init__(self, path: str, metadata: dict, channels: dict):
        self.path = path
        self.metadata = metadata or {}
        self.channels = channels or {}

    def list_channels(self):
        return [(name, info.get("title", name)) for name, info in self.channels.items()]

    def get_channel(self, name):
        return self.channels.get(name)

    def get_metadata(self):
        return self.metadata

    def __repr__(self):
        return f"Sm4Data(path={os.path.basename(self.path)!r}, channels={len(self.channels)})"

class DataManager:
    def __init__(self):
        self.current_data: Optional[MeasurementData] = None
        self.filepath: Optional[str] = None

    def load_file(self, filepath: str) -> MeasurementData:
        try:
            logger.info("Loading file: %s", filepath)
            self.current_data = load_file(filepath)
            self.filepath = filepath
            logger.info("File loaded: %s", filepath)
            return self.current_data
        except Exception:
            logger.exception("Failed to load file: %s", filepath)
            raise

    # ---- Convenience accessors ----
    def get_metadata(self):
        return self.current_data.get_metadata() if self.current_data else {}

    def get_metadata_items(self):
        if not self.current_data:
            return []

        metadata = self.current_data.get_metadata()
        return list(metadata.items())

    def get_channels(self):
        return self.current_data.list_channels() if self.current_data else []

    def get_channel_data(self, name: str):
        return self.current_data.get_channel(name) if self.current_data else None

# ------------------------- Utilities -------------------------
def to_float_safe(val, default=None):
    if val is None:
        return default
    if isinstance(val, str):
        v = val.strip().replace(" ", "").replace(",", "").replace(";", "")
    else:
        v = val
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def infer_channel_type(attrs: Dict[str, Any]) -> str:
    if is_metric_channel({"attrs": attrs}):
        return "topo"
    return "sts"

def is_metric_channel(v: Dict[str, Any]) -> bool:
    a = v.get("attrs", {})
    units = [
        str(a.get("RHK_Xunits", "")).lower().strip(),
        str(a.get("RHK_Yunits", "")).lower().strip(),
        str(a.get("RHK_Zunits", "")).lower().strip(),
    ]
    allowed = {"m", "meter", "metre", "nm", "nanometer", "å", "angstrom", "angstrem"}
    return all(u in allowed for u in units)

def extract_surface_size(attrs: dict) -> str:
    xscale = to_float_safe(attrs.get("RHK_Xscale"))
    yscale = to_float_safe(attrs.get("RHK_Yscale"))
    xsize = to_float_safe(attrs.get("RHK_Xsize"))
    ysize = to_float_safe(attrs.get("RHK_Ysize"))
    xunit = attrs.get("RHK_Xunits") or "m"
    yunit = attrs.get("RHK_Yunits") or "m"

    if not all(isinstance(v, float) for v in (xscale, yscale, xsize, ysize)):
        return "Unknown"

    x_extent = abs(xscale * xsize)
    y_extent = abs(yscale * ysize)

    if xunit == yunit:
        return f"[{x_extent:.1e}, {y_extent:.1e}] {xunit}"
    return f"{x_extent:.1e} {xunit}, {y_extent:.1e} {yunit}"

def extract_bias_range(attrs: dict) -> str:
    bias_start = to_float_safe(attrs.get("RHK_Bias"))
    bias_step = to_float_safe(attrs.get("RHK_Xscale"))
    bias_nstep = to_float_safe(attrs.get("RHK_Xsize"))
    bias_unit = attrs.get("RHK_Xunit") or "V"

    valid = (
        isinstance(bias_start, float) and
        isinstance(bias_step, float) and
        isinstance(bias_nstep, float)
    )
    if not valid:
        return "Unknown"

    bias_end = bias_start + abs(bias_step * bias_nstep)
    if f"{bias_start:.4f}" == f"{bias_end:.4f}":
        return f"{bias_start:.4f} [{bias_unit}], difference = {bias_end-bias_start:.16f}"
    return f"[{bias_start:.4f}, {bias_end:.4f}] {bias_unit}"

# ------------------------- Metadata builder -------------------------
def build_common_metadata(path: str, attrs: dict, channel_type: str, file_type: str = "sm4") -> dict:
    filename = os.path.basename(path)
    title = attrs.get("filename") or filename
    date = attrs.get("RHK_Date", "Unknown")
    comment = attrs.get("RHK_SessionText", "—")

    metadata = {
        "Original": title,
        "Date": date,
        "Surface size (STM)": "Unknown",
        "Bias range (STS)": extract_bias_range(attrs),
        "Session Comment": comment
    }

    # TOPO only - add surface size
    if channel_type == "topo":
        metadata["Surface size (STM)"] = extract_surface_size(attrs)

    # CSV only - add CSV filename
    if file_type == "csv":
        metadata["CSV filename"] = filename

    return metadata

# ------------------------- Channel builders -------------------------
def build_channels_from_sm4(spym_data) -> dict:
    channels = {}

    for var_name, da in spym_data.data_vars.items():
        title = (da.attrs.get("long_name", var_name)).strip()
        ch_type = infer_channel_type(da.attrs)
        data_arr = np.asarray(da.data)

        if data_arr.ndim == 1:  # 2D array expected
            data_arr = data_arr.reshape(-1, 1)
        channels[var_name] = {
            "type": ch_type,
            "title": title,
            "data": data_arr,
            "attrs": dict(da.attrs),
            "file_type": "sm4",
        }
    return channels

def build_channel_from_csv(
    x: np.ndarray,
    y_matrix: np.ndarray,
    metadata: dict,
    y_labels: list[str]
) -> dict:

    # Normalize input
    x_arr = np.asarray(x, dtype=float)
    y_mat = np.asarray(y_matrix, dtype=float)
    attrs = {k: to_float_safe(v, default=v) for k, v in metadata.items()}

    # Prepare title
    raw_title = metadata.get("RHK_Label") or metadata.get("Label") or ""
    title = raw_title.strip() or "Unknown"

    # Ensure minimal fields exist
    if "CSV filename" in metadata and "File" not in attrs:
        attrs["File"] = metadata["CSV filename"]

    if "RHK_Bias" not in attrs and x_arr.size > 0:
        attrs["RHK_Bias"] = float(x_arr[0])

    if "RHK_Xscale" not in attrs and x_arr.size > 1:
        attrs["RHK_Xscale"] = float(x_arr[1] - x_arr[0])

    if "RHK_Xsize" not in attrs:
        attrs["RHK_Xsize"] = int(y_mat.shape[0])

    # Store curve labels
    attrs["CSV_Ylabels"] = list(y_labels)

    # Add default units if missing
    if "RHK_Xunits" not in attrs:
        attrs["RHK_Xunits"] = "V"
    if "RHK_Yunits" not in attrs:
        attrs["RHK_Yunits"] = "a.u."
    if "RHK_Zunits" not in attrs:
        attrs["RHK_Zunits"] = "a.u."

    ch_type = infer_channel_type(attrs)

    # Always 2D array
    if y_mat.ndim == 1:
        y_mat = y_mat.reshape(-1, 1)

    # Standardized channel dict
    return {
        "csv_data": {
            "type": ch_type,
            "title": title,
            "data": y_mat,
            "attrs": attrs,
            "file_type": "csv",
        }
    }

# ------------------------- File loaders -------------------------
def load_sm4_file(filepath: str) -> MeasurementData:
    data = spym.load(filepath)
    channels = build_channels_from_sm4(data)

    topo_var = next((k for k, v in channels.items() if is_metric_channel(v)), None)
    first_var = next(iter(data.data_vars.keys()))

    if topo_var:
        topo_attrs = channels[topo_var]["attrs"]

        sts_attrs = dict(data[first_var].attrs)
    else:
        # No topo channel: use first channel attrs for both topo+sts
        topo_attrs = dict(data[first_var].attrs)
        sts_attrs = topo_attrs

    # Build STS metadata
    metadata = build_common_metadata(
        filepath,
        sts_attrs,
        "sts",
        file_type="sm4",
    )

    if topo_var:
        topo_meta = build_common_metadata(
            filepath,
            topo_attrs,
            "topo",
            file_type="sm4",
        )

        metadata.update({
            key: val
            for key, val in topo_meta.items()
            if "Surface" in key
        })

    return MeasurementData(filepath, metadata, channels)


def load_csv_file(path: str) -> MeasurementData:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 5:
        raise ValueError("CSV file too short or invalid format")

    # Metadata extraction
    meta_keys = rows[0][1:]
    meta_vals = rows[1][1:]
    metadata = dict(zip(meta_keys, meta_vals))
    metadata["CSV filename"] = os.path.basename(path)
    metadata["File"] = metadata.get("File") or metadata["CSV filename"]

    # Infer channel type from title
    title = (str(metadata.get("RHK_Label") or "").strip() or "Topography")
    ch_type = infer_channel_type(metadata)

    # --- TOPO channel parsing ---
    if ch_type == "topo":
        header = rows[3]
        if len(header) < 4:
            raise ValueError("Invalid TOPO CSV format (not enough columns)")

        def parse_label_and_unit(lbl):
            s = str(lbl or "").strip()
            if "[" in s and "]" in s:
                name = s.split("[", 1)[0].strip()
                unit = s.split("[", 1)[1].split("]", 1)[0].strip()
            else:
                name, unit = s, "m"
            return name, unit

        _, x_unit = parse_label_and_unit(header[0])
        _, y_unit = parse_label_and_unit(header[1])

        numeric_rows, x_list, y_list = [], [], []
        for r in rows[4:]:
            if not r or len(r) < 4:
                continue
            z_vals = [to_float_safe(v, default=np.nan) for v in r[3:]]
            if all(np.isnan(z_vals)):
                continue
            x_list.append(to_float_safe(r[0], default=np.nan))
            y_list.append(to_float_safe(r[1], default=np.nan))
            numeric_rows.append(z_vals)

        if not numeric_rows:
            raise ValueError("No numeric rows found for TOPO data")

        z_arr = np.array(numeric_rows, dtype=float)
        x_arr = np.array(x_list, dtype=float)
        y_arr = np.array(y_list, dtype=float)

        # Remove columns that are all NaN
        valid_cols = ~np.all(np.isnan(z_arr), axis=0)
        z_arr = z_arr[:, valid_cols]

        ny, nx = z_arr.shape

        attrs = {k: to_float_safe(v, default=v) for k, v in metadata.items()}
        attrs["RHK_Xsize"] = nx
        attrs["RHK_Ysize"] = ny
        attrs["RHK_Xscale"] = float(np.nanmean(np.diff(np.unique(x_arr)))) if nx > 1 else 1.0
        attrs["RHK_Yscale"] = float(np.nanmean(np.diff(np.unique(y_arr)))) if ny > 1 else 1.0
        attrs["RHK_Xunits"] = attrs.get("RHK_Xunits") or x_unit or "m"
        attrs["RHK_Yunits"] = attrs.get("RHK_Yunits") or y_unit or "m"
        attrs["RHK_Zunits"] = attrs.get("RHK_Zunits") or metadata.get("RHK_Zunits") or "m"

        channel = {
            "csv_topo": {
                "type": "topo",
                "title": title,
                "data": z_arr,
                "attrs": attrs,
            }
        }

        common_meta = build_common_metadata(path, attrs, "topo", file_type="csv")

        return MeasurementData(path, common_meta, channel)

    # --- STS channel parsing ---
    header = rows[3]
    if len(header) < 3:
        raise ValueError("CSV header must contain Index, X and at least one Y column")

    y_labels_raw = header[2:]
    y_labels = [lbl.strip() if str(lbl).strip() else f"C{i+1}" for i, lbl in enumerate(y_labels_raw)]

    x_list, y_rows = [], []
    for r in rows[4:]:
        if not r or len(r) < 3:
            continue
        row_vals = [to_float_safe(v, default=np.nan) for v in r]
        x_list.append(row_vals[1])
        y_rows.append(row_vals[2:])

    x_arr = np.array(x_list, dtype=float)
    y_raw = np.array(y_rows, dtype=float)

    valid_cols = ~np.all(np.isnan(y_raw), axis=0)
    y_arr = y_raw[:, valid_cols]
    y_labels = [lbl for lbl, ok in zip(y_labels, valid_cols) if ok]

    channels = build_channel_from_csv(x_arr, y_arr, metadata, y_labels)
    channel_type = next(iter(channels.values()))["type"]

    common_meta = build_common_metadata(path, channels[next(iter(channels))]["attrs"], channel_type, file_type="csv")

    return MeasurementData(path, common_meta, channels)

# ------------------------- Unified loader -------------------------
def load_file(path: str) -> MeasurementData:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".csv":
        return load_csv_file(path)
    elif ext == ".sm4":
        return load_sm4_file(path)
    raise ValueError(f"Unsupported file type: {ext}")

# ------------------------- Convenience accessors -------------------------
load = load_file
load_sm4 = load_sm4_file
load_csv = load_csv_file

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="sm4_adapter test runner")
    p.add_argument("path", help="Path to .sm4 or .csv file to load")
    args = p.parse_args()
