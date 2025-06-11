# -*- coding: utf-8 -*-
"""
Простой логгер клавиатуры + мыши для ULTRAKILL.
Записывает события в JSON, Ctrl-C = корректное завершение.
"""
import json, time, threading, argparse
from pynput import keyboard, mouse

# -------- аргумент командной строки --------
parser = argparse.ArgumentParser()
parser.add_argument("--outfile", required=True,
                    help="Файл, куда сохранить события (JSON).")
args = parser.parse_args()

# -------- глобальный словарь событий --------
events = {}
lock   = threading.Lock()

def now() -> str:
    """Текущий момент времени с точностью 0.0001 с."""
    return f"{time.time():.4f}"

def _log(update: dict):
    """Потокобезопасно добавляем/обновляем запись."""
    with lock:
        events.setdefault(now(), {}).update(update)

# ---------- обработчики pynput ---------------
def on_press(key):                      _log({str(key): 1})
def on_release(key):                    _log({str(key): 0})
def on_move(x, y):                      _log({"mouse": [x, y]})
def on_click(x, y, button, pressed):    _log({str(button): int(pressed)})

keyboard.Listener(on_press=on_press,
                  on_release=on_release).start()
mouse.Listener(on_move=on_move,
               on_click=on_click).start()

print("[logger] Recording…  жми  Ctrl+C  чтобы остановить")

try:
    while True:
        time.sleep(0.25)
except KeyboardInterrupt:
    pass

# ---------- сохраняем ----------
with lock:
    snapshot = dict(events)

print(f"[logger] Saving {args.outfile}  (events={len(snapshot)})")
with open(args.outfile, "w", encoding="utf-8") as f:
    json.dump(snapshot, f, ensure_ascii=False)
print("[logger] Done.")
