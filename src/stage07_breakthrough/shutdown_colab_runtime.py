#!/usr/bin/env python
"""Flush Drive then delete/unassign the current Colab runtime."""
from __future__ import annotations
import time
try:
    from google.colab import drive
    print("[shutdown] flushing/unmounting Google Drive...",flush=True)
    try:drive.flush_and_unmount()
    except Exception as e:print("[shutdown] Drive flush warning:",repr(e),flush=True)
except Exception as e:
    print("[shutdown] google.colab.drive unavailable:",repr(e),flush=True)

time.sleep(3)
from google.colab import runtime
print("[shutdown] unassigning Colab runtime now.",flush=True)
runtime.unassign()
