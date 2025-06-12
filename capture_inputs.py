#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyboard and mouse logger for ULTRAKILL."""

import argparse
import json
import threading
import time
from pynput import keyboard, mouse

# Shared dictionary of events
EVENTS = {}
LOCK = threading.Lock()


def now() -> str:
    """Return the current timestamp with 0.0001 sec precision."""
    return f"{time.time():.4f}"


def _log(update: dict) -> None:
    """Thread safe event append/merge."""
    with LOCK:
        EVENTS.setdefault(now(), {}).update(update)


def main(outfile: str) -> None:
    """Capture inputs until Ctrl+C and save to *outfile*."""
    k_listener = keyboard.Listener(
        on_press=lambda k: _log({str(k): 1}),
        on_release=lambda k: _log({str(k): 0}),
    )
    m_listener = mouse.Listener(
        on_move=lambda x, y: _log({"mouse": [x, y]}),
        on_click=lambda x, y, btn, p: _log({str(btn): int(p)}),
    )
    k_listener.start()
    m_listener.start()
    print("[logger] Recording…  жми  Ctrl+C  чтобы остановить")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        k_listener.stop()
        m_listener.stop()

    with LOCK:
        snapshot = dict(EVENTS)

    print(f"[logger] Saving {outfile}  (events={len(snapshot)})")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    print("[logger] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfile", required=True, help="Файл, куда сохранить события (JSON)."
    )
    main(parser.parse_args().outfile)
