try:
    import gym
except ImportError:
    import gymnasium as gym

from gym import spaces
import mss
import numpy as np
import cv2
import time
import pydirectinput
import psutil
import pygetwindow as gw
from pynput import keyboard

# remove delays between key events
pydirectinput.PAUSE = 0
try:
    pydirectinput.MINIMUM_DURATION = 0
except AttributeError:
    pass

# binary keys we can press
A_KEYS = [
    'w','a','s','d',
    'space','shift','ctrl',
    'left','right',
    '1','2','3','4','5',
    'r','f','g'
]

# indices for some common keys
IDX_SPACE = A_KEYS.index('space')
IDX_SHIFT = A_KEYS.index('shift')
IDX_CTRL  = A_KEYS.index('ctrl')
SLOT_OFF  = 9  # index of key '1'

# base resolution used to scale HUD coordinates
BASE_W, BASE_H = 1024, 768

# coordinates of HP bar and style meter using the base resolution
HP_TL, HP_BR   = (80, 632),  (276, 696)
STYLE_TL, STYLE_BR = (780, 230), (1020, 265)


class UltraKillEnv:
    """Very small environment to control ULTRAKILL."""

    def __init__(self, res=(1024, 768), mouse=False, mouse_scale=30, debug=False):
        self.res = res
        self.mouse = mouse
        self.mouse_scale = mouse_scale
        self.debug = debug

        w, h = res
        self.observation_space = spaces.Box(0, 255, shape=(3, h, w), dtype=np.uint8)
        if mouse:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(len(A_KEYS)+2,), dtype=np.float32)
        else:
            self.action_space = spaces.MultiBinary(len(A_KEYS))

        self.sct = mss.mss()
        self.win_bbox = None
        self._update_window()
        self._prepare_regions()

        self._exit = False
        self.listener = keyboard.GlobalHotKeys({'<ctrl>+<alt>+x': self._set_exit})
        self.listener.start()

        self._reset_prev()

    # ------------ helpers ------------
    def _grab(self):
        self._ensure_process()
        self._ensure_window()
        self._check_exit()
        frame = np.asarray(self.sct.grab(self.win_bbox))[:, :, :3]
        return cv2.resize(frame, self.res, interpolation=cv2.INTER_AREA)

    def _ensure_process(self):
        for proc in psutil.process_iter(["name"]):
            if proc.info.get("name", "").lower() == "ultrakill.exe":
                return
        raise RuntimeError("ULTRAKILL.exe process not found")

    def _ensure_window(self):
        if self.win_bbox is None:
            self._update_window()

    def _update_window(self):
        wins = [w for w in gw.getAllTitles() if "ULTRAKILL" in w.upper()]
        if not wins:
            raise RuntimeError("ULTRAKILL window not found")
        win = gw.getWindowsWithTitle(wins[0])[0]
        self.win_bbox = {
            'left': win.left,
            'top': win.top,
            'width': win.width,
            'height': win.height,
        }

    def _prepare_regions(self):
        self.hp_box = self._scale_coords(HP_TL, HP_BR)
        self.style_box = self._scale_coords(STYLE_TL, STYLE_BR)

    def _scale_coords(self, tl, br):
        w, h = self.res
        x1 = int(tl[0] * w / BASE_W)
        y1 = int(tl[1] * h / BASE_H)
        x2 = int(br[0] * w / BASE_W)
        y2 = int(br[1] * h / BASE_H)
        return x1, y1, x2, y2

    def _crop_box(self, frame, box):
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]

    def _set_exit(self):
        self._exit = True

    def _check_exit(self):
        if getattr(self, '_exit', False):
            if hasattr(self, 'listener'):
                self.listener.stop()
            raise SystemExit('Exit hotkey pressed')

    def _reset_prev(self):
        self.prev_hp = 1.0
        self.prev_style = 0
        self.prev_frame = None

    # ------------ Gym API ------------
    def reset(self, *, seed=None, options=None):
        self._ensure_process()
        self._reset_prev()
        frame = self._grab()
        return frame.transpose(2, 0, 1), {}

    def step(self, action):
        self._ensure_process()
        dx = dy = 0
        if self.mouse and len(action) >= len(A_KEYS)+2:
            dx = int(float(action[len(A_KEYS)]) * self.mouse_scale)
            dy = int(float(action[len(A_KEYS)+1]) * self.mouse_scale)

        for i, key in enumerate(A_KEYS):
            pressed = action[i] > 0
            if key in ('left', 'right'):
                btn = 'left' if key == 'left' else 'right'
                (pydirectinput.mouseDown if pressed else pydirectinput.mouseUp)(button=btn)
            else:
                (pydirectinput.keyDown if pressed else pydirectinput.keyUp)(key)

        if dx or dy:
            pydirectinput.moveRel(dx, dy)

        time.sleep(0.016)  # roughly 60 FPS
        frame = self._grab()
        obs = frame.transpose(2, 0, 1)

        hp_region = self._crop_box(frame, self.hp_box)
        hp = hp_region[:, :, 2].mean() / 255

        style_region = self._crop_box(frame, self.style_box)
        style = (style_region > 220).sum()

        hp_loss = self.prev_hp - hp
        style_gain = style - self.prev_style

        reward = style_gain * 0.25 - hp_loss * 5.0
        terminated = hp <= 0.01

        self.prev_hp = hp
        self.prev_style = style
        self.prev_frame = frame

        if self.debug:
            print(f"reward {reward:.2f} hp {hp:.2f} style {style_gain}")

        return obs, reward, terminated, False, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()


# simple Gym wrapper so SB3 can use it
class UltraKillWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mouse=False, mouse_scale=30, debug=False):
        super().__init__()
        self.core = UltraKillEnv(mouse=mouse, mouse_scale=mouse_scale, debug=debug)
        self.action_space = self.core.action_space
        self.observation_space = self.core.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.core.reset()
        return obs, info

    def step(self, action):
        return self.core.step(action)

    def render(self, *args, **kwargs):
        return None

    def close(self):
        self.core.close()

