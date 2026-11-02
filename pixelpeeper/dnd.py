"""Native Windows drag-and-drop for Tkinter (WM_DROPFILES), no extra packages.

The Tk window procedure is subclassed with ctypes.  The callback only parses the
file list and pushes it into a queue; all Tk calls happen later on the main
thread, which keeps Tk's interpreter state safe.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import queue
import sys

WM_DROPFILES = 0x0233
GWLP_WNDPROC = -4
GMEM_MOVEABLE = 0x0002
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


class _DROPFILES(ctypes.Structure):
    _fields_ = [("pFiles", wt.DWORD), ("pt", wt.POINT), ("fNC", wt.BOOL), ("fWide", wt.BOOL)]


class DropTarget:
    """Attach to a Tk toplevel; dropped paths appear in ``.queue``."""

    def __init__(self, widget):
        self.widget = widget
        self.queue: "queue.Queue[list[str]]" = queue.Queue()
        self.available = False
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._old = None
        self._proc_ref = None
        if sys.platform != "win32":
            return
        try:
            self._bind()
            self.available = True
        except Exception:
            self.available = False

    def _bind(self):
        u, s, k = self._user32, self._shell32, self._kernel32
        u.SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_void_p]
        u.SetWindowLongPtrW.restype = ctypes.c_void_p
        u.CallWindowProcW.argtypes = [ctypes.c_void_p, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
        u.CallWindowProcW.restype = LRESULT
        s.DragAcceptFiles.argtypes = [wt.HWND, wt.BOOL]
        s.DragAcceptFiles.restype = None
        s.DragQueryFileW.argtypes = [wt.HANDLE, wt.UINT, wt.LPWSTR, wt.UINT]
        s.DragQueryFileW.restype = wt.UINT
        s.DragFinish.argtypes = [wt.HANDLE]
        s.DragFinish.restype = None
        k.GlobalAlloc.argtypes = [wt.UINT, ctypes.c_size_t]
        k.GlobalAlloc.restype = wt.HGLOBAL
        k.GlobalLock.argtypes = [wt.HGLOBAL]
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalUnlock.argtypes = [wt.HGLOBAL]

        self._hwnd = self.widget.winfo_id()
        self._proc_ref = WNDPROC(self._proc)
        s.DragAcceptFiles(self._hwnd, True)
        self._old = u.SetWindowLongPtrW(self._hwnd, GWLP_WNDPROC,
                                       ctypes.cast(self._proc_ref, ctypes.c_void_p))

    def _proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_DROPFILES:
            paths: list[str] = []
            try:
                count = self._shell32.DragQueryFileW(wparam, 0xFFFFFFFF, None, 0)
                for i in range(count):
                    n = self._shell32.DragQueryFileW(wparam, i, None, 0)
                    buf = ctypes.create_unicode_buffer(n + 1)
                    self._shell32.DragQueryFileW(wparam, i, buf, n + 1)
                    paths.append(buf.value)
            except Exception:
                pass
            finally:
                try:
                    self._shell32.DragFinish(wparam)
                except Exception:
                    pass
            if paths:
                self.queue.put(paths)
            return 0
        return self._user32.CallWindowProcW(self._old, hwnd, msg, wparam, lparam)

    def close(self):
        if self._old and self.available:
            try:
                self._user32.SetWindowLongPtrW(self._hwnd, GWLP_WNDPROC, self._old)
            except Exception:
                pass
            self._old = None
