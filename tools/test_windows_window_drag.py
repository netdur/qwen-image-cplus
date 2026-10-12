"""Exercise native window dragging with real mouse input on a Windows desktop.

Usage: python tools/test_windows_window_drag.py path/to/gui.exe
Requires an interactive desktop; the app is launched and closed by this test.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import time


def run(exe):
    u = c.WinDLL("user32", use_last_error=True)
    u.SetProcessDpiAwarenessContext(c.c_void_p(-4))
    u.GetWindowThreadProcessId.argtypes = [w.HWND, c.POINTER(w.DWORD)]
    u.GetWindowRect.argtypes = [w.HWND, c.POINTER(w.RECT)]
    u.ClientToScreen.argtypes = [w.HWND, c.POINTER(w.POINT)]
    u.GetDpiForWindow.argtypes = [w.HWND]
    u.SetWindowPos.argtypes = [w.HWND, w.HWND, c.c_int, c.c_int, c.c_int, c.c_int, w.UINT]
    u.SetForegroundWindow.argtypes = [w.HWND]
    u.GetForegroundWindow.restype = w.HWND
    u.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
    enum_proc = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    u.IsWindowVisible.argtypes = [w.HWND]
    u.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, c.c_int]
    u.IsZoomed.argtypes = [w.HWND]
    u.ShowWindow.argtypes = [w.HWND, c.c_int]

    def rect(hwnd):
        r = w.RECT()
        assert u.GetWindowRect(hwnd, c.byref(r))
        return [r.left, r.top, r.right, r.bottom]

    def move(x, y):
        u.mouse_event(0x8001, round(x * 65535 / (u.GetSystemMetrics(0) - 1)),
                      round(y * 65535 / (u.GetSystemMetrics(1) - 1)), 0, 0)
        time.sleep(.3)

    exe = Path(exe).resolve()
    with exe.with_suffix(".drag.log").open("w", encoding="utf-8") as log:
        p = subprocess.Popen([str(exe)], cwd=exe.parent, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        hwnd = None
        try:
            deadline = time.monotonic() + 30
            while not hwnd and time.monotonic() < deadline:
                found = []

                def visit(h, _):
                    pid = w.DWORD()
                    u.GetWindowThreadProcessId(h, c.byref(pid))
                    title = c.create_unicode_buffer(256)
                    u.GetWindowTextW(h, title, 256)
                    if pid.value == p.pid and title.value == "Qwen Image" and u.IsWindowVisible(h):
                        found.append(h)
                    return True

                u.EnumWindows(enum_proc(visit), 0)
                hwnd = found[0] if found else None
                assert p.poll() is None, "App exited; inspect its .drag.log"
                time.sleep(.1)
            assert hwnd, "No Qwen Image window found on this desktop"
            assert u.SetWindowPos(hwnd, None, 80, 80, 0, 0, 0x0045)
            u.SetForegroundWindow(hwnd)
            time.sleep(1)
            assert u.GetForegroundWindow() == hwnd, "Test window must be foreground"
            scale = u.GetDpiForWindow(hwnd) / 96
            def point(x, y):
                origin = w.POINT(0, 0)
                assert u.ClientToScreen(hwnd, c.byref(origin))
                return origin.x + round(x * scale), origin.y + round(y * scale)

            def click():
                u.mouse_event(2, 0, 0, 0, 0)
                time.sleep(.05)
                u.mouse_event(4, 0, 0, 0, 0)

            for name, x, y in [("compose header", 250, 26),
                               ("brand", 440, 22), ("preview header", 700, 32)]:
                assert u.SetWindowPos(hwnd, None, 80, 80, 0, 0, 0x0045)
                time.sleep(.3)
                sx, sy = point(x, y)
                move(sx, sy)
                before = rect(hwnd)
                u.mouse_event(2, 0, 0, 0, 0)
                time.sleep(.3)
                move(sx + 100, sy + 70)
                held = rect(hwnd)
                u.mouse_event(4, 0, 0, 0, 0)
                time.sleep(.3)
                released = rect(hwnd)
                move(sx + 160, sy + 110)
                after = rect(hwnd)
                evidence = dict(region=name, before=before, held=held, released=released, after=after)
                print(json.dumps(evidence), flush=True)
                assert abs(held[0] - before[0] - 100) <= 3, "Window did not follow horizontal movement while held"
                assert abs(held[1] - before[1] - 70) <= 3, "Window did not follow vertical movement while held"
                assert after == released, "Window kept moving after release"

            move(*point(250, 26))
            click()
            click()
            time.sleep(.5)
            assert u.IsZoomed(hwnd), "Double-clicking the caption did not maximize"
            u.ShowWindow(hwnd, 9)
            time.sleep(.5)
            # The custom window-button group starts at x=16, with 46-DIP buttons.
            move(*point(85, 26))
            click()
            time.sleep(.5)
            assert u.IsZoomed(hwnd), "Maximize button was swallowed by the caption region"
            move(*point(85, 26))
            click()
            time.sleep(.5)
            assert not u.IsZoomed(hwnd), "Restore button did not restore"
            # Content below the title bar must continue to receive client input.
            sx, sy = point(700, 100)
            move(sx, sy)
            before = rect(hwnd)
            u.mouse_event(2, 0, 0, 0, 0)
            time.sleep(.2)
            move(sx + 80, sy + 40)
            u.mouse_event(4, 0, 0, 0, 0)
            assert rect(hwnd) == before, "Preview content became a drag region"
            print("PASS: all headers drag and stop on release; double-click and window buttons work", flush=True)
        except Exception:
            if hwnd:
                try:
                    from PIL import ImageGrab
                    ImageGrab.grab(bbox=tuple(rect(hwnd))).save(exe.with_suffix(".drag.png"))
                except (ImportError, OSError):
                    pass
            raise
        finally:
            u.mouse_event(4, 0, 0, 0, 0)
            # End a broken move loop before asking the test instance to close.
            u.keybd_event(27, 0, 0, 0)
            u.keybd_event(27, 0, 2, 0)
            if hwnd:
                u.PostMessageW(hwnd, 0x0010, 0, 0)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        assert p.returncode == 0, "App did not close cleanly; inspect its .drag.log"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe")
    run(parser.parse_args().exe)
