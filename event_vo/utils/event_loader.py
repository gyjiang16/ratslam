"""
Event data loading from CSV, NPZ, and AEDAT4 formats.

All formats are normalized to a unified NumPy array with columns:
    [timestamp_us, x, y, polarity]

Supported CSV formats:
  - Standard:    t,x,y,p         (timestamp in microseconds, polarity ±1)
  - DV software: timestamp,x,y,polarity  (polarity 0/1, timestamp in us)
  - Flexible:    auto-detects column order from header names

NPZ: structured array with fields 't','x','y','p' or plain array
AEDAT4: uses dv-processing (iniVation SDK)
"""

import numpy as np
from pathlib import Path
import sys


_COL_ALIASES = {
    "t": ["t", "timestamp", "ts", "time"],
    "x": ["x", "col", "column"],
    "y": ["y", "row"],
    "p": ["p", "pol", "polarity", "on_off"],
}


def load_events(filepath: str) -> np.ndarray:
    """
    Load events and return as (N, 4) float64 array: [t_us, x, y, polarity].
    Polarity is normalized to +1/-1 regardless of input format (0/1 or ±1).
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".csv" or ext == ".txt":
        events = _load_csv(path)
    elif ext == ".npz":
        events = _load_npz(path)
    elif ext == ".aedat4":
        events = _load_aedat4(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    pol = events[:, 3]
    if np.all((pol == 0) | (pol == 1)):
        events[:, 3] = pol * 2 - 1

    events = events[events[:, 0].argsort()]

    return events


def _detect_columns(header_line: str):
    """
    Parse a CSV header to determine which column index maps to t, x, y, p.
    Returns a dict {field: col_index} or None if no header detected.
    """
    parts = [h.strip().lower() for h in header_line.split(",")]

    mapping = {}
    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            for i, col_name in enumerate(parts):
                if col_name == alias:
                    mapping[field] = i
                    break
            if field in mapping:
                break

    if len(mapping) >= 3:
        return mapping
    return None


def _fast_load_numeric_csv(path, delimiter=",", skiprows=0) -> np.ndarray:
    """Load numeric CSV into float64 ndarray. Uses pandas C engine when available."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=delimiter, skiprows=skiprows,
                         header=None, dtype=np.float64,
                         na_filter=False, encoding="utf-8-sig")
        return df.values
    except ImportError:
        print("  (tip: 'pip install pandas' for 5-10x faster CSV loading)",
              file=sys.stderr)
        return np.loadtxt(path, delimiter=delimiter,
                          skiprows=skiprows, dtype=np.float64)


def _load_csv(path: Path) -> np.ndarray:
    """
    Load CSV with flexible column detection.
    Handles: t,x,y,p | timestamp,x,y,polarity | headerless (assumed t,x,y,p)
    """
    def _is_probably_numeric_row(line: str, delim: str) -> bool:
        parts = [p.strip() for p in line.split(delim) if p.strip() != ""]
        if not parts:
            return False
        try:
            float(parts[0])
            return True
        except ValueError:
            return False

    first_nonempty = ""
    leading_skip = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if s == "" or s.startswith("#"):
                leading_skip += 1
                continue
            first_nonempty = s
            break

    if first_nonempty == "":
        raise ValueError(f"CSV appears empty: {path}")

    delimiter = "," if "," in first_nonempty else ("\t" if "\t" in first_nonempty else ",")
    col_map = _detect_columns(first_nonempty.replace("\t", ",") if delimiter == "\t" else first_nonempty)

    if col_map is not None:
        data = _fast_load_numeric_csv(path, delimiter, leading_skip + 1)
        t_col = col_map.get("t", 0)
        x_col = col_map.get("x", 1)
        y_col = col_map.get("y", 2)
        p_col = col_map.get("p", 3) if "p" in col_map else None
    else:
        is_numeric = _is_probably_numeric_row(first_nonempty, delimiter)
        skiprows = leading_skip + (0 if is_numeric else 1)
        data = _fast_load_numeric_csv(path, delimiter, skiprows)
        t_col, x_col, y_col = 0, 1, 2
        p_col = 3 if data.shape[1] >= 4 else None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    t = data[:, t_col]
    x = data[:, x_col]
    y = data[:, y_col]
    p = data[:, p_col] if p_col is not None else np.ones(len(data))

    return np.column_stack([t, x, y, p])


def _load_npz(path: Path) -> np.ndarray:
    """Load NPZ — handles structured and plain arrays."""
    with np.load(path, allow_pickle=True) as f:
        keys = list(f.keys())
        arr = f[keys[0]]

    if arr.dtype.names is not None:
        names = arr.dtype.names
        t_field = next((n for n in names if n.lower() in ("t", "timestamp", "ts")), names[0])
        x_field = next((n for n in names if n.lower() in ("x", "col")), names[1] if len(names) > 1 else None)
        y_field = next((n for n in names if n.lower() in ("y", "row")), names[2] if len(names) > 2 else None)
        p_field = next((n for n in names if n.lower() in ("p", "pol", "polarity")), None)

        t = arr[t_field].astype(np.float64)
        x = arr[x_field].astype(np.float64) if x_field else np.zeros_like(t)
        y = arr[y_field].astype(np.float64) if y_field else np.zeros_like(t)
        p = arr[p_field].astype(np.float64) if p_field else np.ones_like(t)
        return np.column_stack([t, x, y, p])

    arr = arr.astype(np.float64)
    if arr.shape[1] >= 4:
        return arr[:, :4]
    elif arr.shape[1] == 3:
        p = np.ones((arr.shape[0], 1), dtype=np.float64)
        return np.hstack([arr, p])
    else:
        raise ValueError(f"NPZ array must have >=3 columns, got {arr.shape[1]}")


def _load_aedat4(path: Path) -> np.ndarray:
    """
    Load AEDAT4 using dv-processing (iniVation SDK).
    This is the native format for DVXplorer Mini recorded via DV software.
    """
    try:
        import dv_processing as dv
    except ImportError:
        raise ImportError(
            "AEDAT4 format requires dv-processing.\n"
            "Install: pip install dv-processing\n"
            "Or export to CSV from DV software."
        )

    reader = dv.io.MonoCameraRecording(str(path))

    all_t, all_x, all_y, all_p = [], [], [], []

    while reader.isEventStreamAvailable():
        batch = reader.getNextEventBatch()
        if batch is None:
            break
        events_np = np.array(batch.numpy())
        if len(events_np) == 0:
            continue
        all_t.append(events_np["timestamp"].astype(np.float64))
        all_x.append(events_np["x"].astype(np.float64))
        all_y.append(events_np["y"].astype(np.float64))
        all_p.append(events_np["polarity"].astype(np.float64))

    if not all_t:
        raise ValueError("No events found in AEDAT4 file")

    t = np.concatenate(all_t)
    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)

    return np.column_stack([t, x, y, p])


def get_time_range(events: np.ndarray):
    """Return (t_start, t_end, duration) in microseconds."""
    t_start = events[0, 0]
    t_end = events[-1, 0]
    return t_start, t_end, t_end - t_start
