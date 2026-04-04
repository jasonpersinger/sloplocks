# ISOMount — Design Spec

**Date:** 2026-04-04  
**Status:** Approved

## Overview

A simple Linux desktop application for mounting and unmounting ISO files. Built with Python + GTK4. Uses `udisksctl` as the backend so no `sudo` is required.

## Features

- Browse for and mount an ISO file
- View all currently mounted ISOs in a list (filename + mount point)
- Unmount individual ISOs
- Open a mount point in the file manager
- System tray icon; closing the window minimizes to tray rather than quitting
- Pre-populates the list on startup by detecting ISOs already mounted via loop devices

## Architecture

Two-layer design:

- **Backend** (`isomount/backend.py`) — wraps `udisksctl` subprocess calls. No GTK imports. All mount/unmount/list logic lives here, making it independently testable.
- **GUI** (`isomount/app.py`) — GTK4 window and tray icon. Calls the backend, updates the list view in response.
- **Entry point** (`isomount/__main__.py`) — launches the GTK main loop, enforces single instance via a lock file (`/tmp/isomount.lock`).

## File Structure

```
isomount/
├── __main__.py       ← entry point, single-instance lock
├── app.py            ← GTK4 window, tray icon, signal wiring
├── backend.py        ← udisksctl wrapper (mount/unmount/list)
└── resources/
    └── icon.png      ← tray/window icon
pyproject.toml        ← deps: PyGObject
tests/
└── test_backend.py
```

## UI Layout

Single window (~480px wide):

1. **Top bar** — `Gtk.Entry` (ISO path, read-only), Browse button (opens `Gtk.FileChooserDialog` filtered to `.iso`), Mount button
2. **List** — `Gtk.ListBox`, one row per mounted ISO:
   - Filename (bold) + mount point (secondary text)
   - Open button (launches `xdg-open <mount_point>`)
   - Unmount button
3. **Status bar** — count of mounted ISOs, backend label

## Data Flow

### Mount
1. User selects ISO via file chooser → path populates entry
2. Mount clicked → `backend.mount_iso(path)`
   - `udisksctl loop-setup -f <path>` → returns `/dev/loopX`
   - `udisksctl mount -b /dev/loopX` → returns mount point
3. New row appended to list

### Unmount
1. Unmount clicked on a row → `backend.unmount_iso(loop_dev)`
   - `udisksctl unmount -b /dev/loopX`
   - `udisksctl loop-delete -b /dev/loopX`
2. Row removed from list

### Startup population
- Parse `udisksctl dump` output for loop devices backed by `.iso` files
- Pre-populate list with any found

## Tray Behavior

- Window close button hides the window (app keeps running)
- Tray icon right-click menu: **Show Window**, **Quit**
- Quit does NOT unmount ISOs — user manages them explicitly

## Error Handling

All backend errors (mount failure, file not found, device busy) surface as a `Gtk.MessageDialog`. The app never crashes on a failed operation.

## Backend Interface

```python
# backend.py public interface
def mount_iso(path: str) -> tuple[str, str]:
    """Returns (loop_device, mount_point). Raises RuntimeError on failure."""

def unmount_iso(loop_device: str) -> None:
    """Raises RuntimeError on failure."""

def list_mounted_isos() -> list[dict]:
    """Returns list of {loop_device, iso_path, mount_point} dicts."""
```

## Testing

`tests/test_backend.py` mocks `subprocess.run` and covers:
- Successful mount (parses loop device from udisksctl output)
- Mount failure (raises RuntimeError)
- Successful unmount
- `list_mounted_isos` parses udisksctl dump correctly
- `list_mounted_isos` returns empty list when no loop devices present

No GUI tests — GTK is not headless-friendly and all meaningful logic lives in the backend.

## Dependencies

- Python 3.11+
- `PyGObject` (GTK4 bindings) — system package (`python-gobject`)
- `libayatana-appindicator` — optional, for better tray support on non-GNOME DEs. If unavailable, fall back to `Gtk.StatusIcon` (deprecated but functional). If neither works, the app still runs but close button quits instead of minimizing to tray.

## Non-Goals

- Cross-platform support
- Mounting non-ISO formats
- Persistent mount history
- Auto-mount on file association
