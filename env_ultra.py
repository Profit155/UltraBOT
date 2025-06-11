try:
    import gym                           # старый OpenAI Gym
except ImportError:
    import gymnasium as gym              # если gym нет, берём gymnasium

from gym import spaces
import numpy as np, cv2, mss, time, pydirectinput
import psutil
import pygetwindow as gw
from pynput import keyboard
# ─────────── action-space (17 бинарных кнопок) ───────────
A_KEYS = [
    'w','a','s','d',          # 0-3  движение
    'space','shift','ctrl',   # 4-6  jump / dash / slide
    'left','right',           # 7-8  LMB / RMB
    '1','2','3','4','5',      # 9-13 weapon slots
    'r','f','g'               # 14-16 hook / parry / knuckle
]

IDX_SPACE = A_KEYS.index('space')
IDX_SHIFT = A_KEYS.index('shift')
IDX_CTRL  = A_KEYS.index('ctrl')
SLOT_OFF  = 9                        # индекс первого слота '1'
MOUSE_AXES = 2                       # dx,dy appended when mouse=True

# Base resolution for scaling HUD coordinates
BASE_W, BASE_H = 1024, 768

# HUD element coordinates (top-left and bottom-right) using base resolution
HP_TL,   HP_BR   = (80, 632),  (276, 696)
STAM_TL, STAM_BR = (80, 664),  (276, 729)
RAIL_TL, RAIL_BR = (80, 725),  (276, 755)
STYLE_TEXT_TL, STYLE_TEXT_BR = (780, 155), (950, 210)
STYLE_BAR_TL,  STYLE_BAR_BR  = (780, 230), (1020, 265)

# Style rank colors in RGB space
STYLE_COLORS = {
    "DESTRUCTIVE": (0, 110, 255),
    "CHAOTIC": (50, 255, 50),
    "BRUTAL": (255, 235, 30),
    "ANARCHIC": (255, 140, 30),
    "SUPREME": (220, 0, 0),
    "SSADISTIC": (220, 0, 0),
    "SSSHITSTORM": (220, 0, 0),
    "ULTRAKILL": (255, 220, 70),

# Approximate BGR colors for style ranks
STYLE_COLORS = {
    "DESTRUCTIVE": (255, 110, 0),
    "CHAOTIC": (50, 255, 50),
    "BRUTAL": (30, 235, 255),
    "ANARCHIC": (30, 140, 255),
    "SUPREME": (0, 0, 220),
    "SSADISTIC": (0, 0, 220),
    "SSSHITSTORM": (0, 0, 220),
    "ULTRAKILL": (70, 220, 255),
}

# Rewards when a new style rank appears
STYLE_REWARD = {
    "ULTRAKILL": 20.0,
    "SSSHITSTORM": 15.0,
    "SSADISTIC": 10.0,
    "SUPREME": 5.0,
    "ANARCHIC": 0.0,
    "BRUTAL": -1.0,
    "CHAOTIC": -1.0,
    "DESTRUCTIVE": -1.0,
}

# ───────────
class UltraKillEnv:
    def __init__(self, res=(1024,768), mouse=False, mouse_scale=30):
        """Initialize env.

        Parameters
        ----------
        res : tuple
            Resolution of grabbed screen.
        mouse : bool
            If True, the action space includes relative mouse movement
            (dx, dy) in range [-1,1].
        mouse_scale : int
            Pixel multiplier for mouse movement per step.
        """
        self.res = res
        self.mouse = mouse
        self.mouse_scale = mouse_scale
        # orientation penalty counters
        self.frames_look_down = 0
        self.frames_look_up   = 0

        w, h = res
        self.observation_space = spaces.Box(
            0, 255, shape=(3, h, w), dtype=np.uint8
        )
        if mouse:
            self.action_space = spaces.Box(-1.0, 1.0,
                                          shape=(len(A_KEYS)+2,),
                                          dtype=np.float32)
        else:
            self.action_space = spaces.MultiBinary(len(A_KEYS))

        self.sct = mss.mss()
        # keep original monitor handle for backward compatibility
        self.mon = self.sct.monitors[1]
        self.win_bbox = None
        self._update_window()

        self._exit = False
        self.listener = keyboard.GlobalHotKeys(
            {'<ctrl>+<alt>+x': self._set_exit}
        )
        self.listener.start()

        # prev-состояние для reward
        self._reset_prev()
        self.frames_since_slot = 0
        self.prev_slot         = None

        self._ensure_process()

    # ---------------- utils ----------------
    def _grab(self):
        self._ensure_process()
        self._ensure_window()
        self._check_exit()
        scr = np.asarray(self.sct.grab(self.win_bbox))[:, :, :3]
        scr = cv2.cvtColor(scr, cv2.COLOR_BGR2RGB)
        scr = np.asarray(self.sct.grab(self.win_bbox))[:, :, :3]  # BGR
        scr = np.asarray(self.sct.grab(self.mon))[:,:,:3]         # BGR
        return cv2.resize(scr, self.res, interpolation=cv2.INTER_AREA)

    def _ensure_process(self):
        """Raise RuntimeError if ULTRAKILL.exe is not running."""
        self._check_exit()
        for proc in psutil.process_iter(["name"]):
            if proc.info.get("name", "").lower() == "ultrakill.exe":
                return
        raise RuntimeError("ULTRAKILL.exe process not found")

    def _ensure_window(self):
        """Ensure the cached bounding box is available."""
        if self.win_bbox is None:
            self._update_window()

    def _update_window(self):
        """Cache the coordinates of the ULTRAKILL window."""
        wins = [w for w in gw.getAllTitles() if "ULTRAKILL" in w.upper()]
        if not wins:
            raise RuntimeError("ULTRAKILL window not found")
        win = gw.getWindowsWithTitle(wins[0])[0]
        self.win_bbox = {
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height,
        }

    def _scale_coords(self, tl, br):
        """Return scaled x1,y1,x2,y2 for the current resolution."""
        w, h = self.res
        x1 = int(tl[0] * w / BASE_W)
        y1 = int(tl[1] * h / BASE_H)
        x2 = int(br[0] * w / BASE_W)
        y2 = int(br[1] * h / BASE_H)
        return x1, y1, x2, y2

    def _crop(self, frame, tl, br):
        """Crop HUD region from frame using base coordinates."""
        x1, y1, x2, y2 = self._scale_coords(tl, br)
        return frame[y1:y2, x1:x2]

    def _set_exit(self):
        self._exit = True

    def _check_exit(self):
        if getattr(self, "_exit", False):
            if hasattr(self, "listener"):
                self.listener.stop()
            raise SystemExit("Exit hotkey pressed")

    def _reset_prev(self):
        self.prev_hp = 1.0
        self.prev_dash = self.prev_rail = 1.0
        self.prev_style = 0
        self.prev_flash = False
        self.prev_dead = False
        self.prev_frame = None            # previous observation frame
        self.stuck_frames = 0             # number of frames with little change
        self.checkpoint_active = False    # checkpoint text currently visible
        self.prev_slot_pressed = [False]*5
        self.frames_since_style = 0       # frames since last style gain
        self.rank_seen = False            # scoreboard rank not yet processed
        self.frames_look_down = 0
        self.frames_look_up   = 0
        self.prev_style_rank = None

    # ---------------- Gym API --------------
    def reset(self, *, seed=None, options=None):
        self._ensure_process()
        self._reset_prev()
        self.frames_since_slot = 0
        self.prev_slot = None
        return self._grab().transpose(2,0,1), {}

    def step(self, action):
        """Send actions and compute reward."""
        self._ensure_process()
        # --- клавиши и мышь ---
        dx = dy = 0
        if self.mouse and len(action) >= len(A_KEYS)+2:
            dx = int(float(action[len(A_KEYS)])   * self.mouse_scale)
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

        time.sleep(0.016)                              # ~60 FPS

        # --- наблюдение ---
        frame = self._grab()
        obs   = frame.transpose(2,0,1)
        w, h  = self.res
        x1, y1, x2, y2 = self._scale_coords(HP_TL, HP_BR)
        hp = frame[y1:y2, x1:x2, 0].mean() / 255
        x1, y1, x2, y2 = self._scale_coords(STAM_TL, STAM_BR)
        dash = frame[y1:y2, x1:x2, 2].mean() / 255
        hp = frame[y1:y2, x1:x2, 2].mean() / 255
        x1, y1, x2, y2 = self._scale_coords(STAM_TL, STAM_BR)
        dash = frame[y1:y2, x1:x2, 0].mean() / 255
        x1, y1, x2, y2 = self._scale_coords(RAIL_TL, RAIL_BR)
        rail = frame[y1:y2, x1:x2, 1].mean() / 255
        style_region = self._crop(frame, STYLE_BAR_TL, STYLE_BAR_BR)
        style = (style_region > 200).sum()
        rank_patch = self._crop(frame, STYLE_TEXT_TL, STYLE_TEXT_BR)
        avg_color = rank_patch.mean(axis=(0, 1))
        style_rank = min(STYLE_COLORS, key=lambda k: np.linalg.norm(avg_color - np.array(STYLE_COLORS[k])))
        flash = frame.mean() > 240
        dark  = frame.mean() < 30
        words = (frame[int(h*30/84):int(h*60/84), int(w*20/84):int(w*64/84), :] > 200).mean() > 0.02
        dead  = dark and words
        checkpoint = (frame[int(h*24/84):int(h*56/84), int(w*18/84):int(w*66/84), :] > 230).mean() > 0.05

        # --- vertical look estimate via brightness profile ---
        vert_profile = frame.mean(axis=2).mean(axis=1)
        band = max(1, int(h * 0.1))
        top_b = vert_profile[:band].mean()
        bot_b = vert_profile[-band:].mean()
        if bot_b - top_b > 5:
            self.frames_look_down += 1
            self.frames_look_up = 0
        elif top_b - bot_b > 5:
            self.frames_look_up += 1
            self.frames_look_down = 0
        else:
            self.frames_look_down = 0
            self.frames_look_up = 0

        # --- reward ---
        r = 0.0

        # exploration based on frame difference
        if self.prev_frame is not None:
            diff = np.mean(np.abs(frame - self.prev_frame))
            if diff < 1.0:
                self.stuck_frames += 1
                if self.stuck_frames > 180:
                    r -= 0.2                     # penalty for staying still
            else:
                r += diff / 50.0                # encourage movement
                if diff > 20.0:
                    r += 1.0                     # likely entered new area
                self.stuck_frames = 0

        # checkpoint detection (white "CHECKPOINT" text)
        if checkpoint and not self.checkpoint_active:
            r += 10.0
            self.checkpoint_active = True
        elif not checkpoint:
            self.checkpoint_active = False

        # end-level rank detection (big letter in center)
        rank_area = frame[int(h*30/84):int(h*54/84), int(w*28/84):int(w*56/84), :]
        rank_brightness = rank_area.mean()
        if rank_brightness > 170 and not getattr(self, "rank_seen", False):
            if rank_brightness > 240:
                r += 50.0  # P rank
            elif rank_brightness > 210:
                r += 30.0  # S or A
            elif rank_brightness > 190:
                r += 10.0  # B
            else:
                r -= 5.0   # C or worse
            self.rank_seen = True
        hp_loss = self.prev_hp - hp
        if hp_loss > 0:
            r -= hp_loss * 5.0                      # сильное наказание за урон
        if hp < 0.1 and self.prev_hp >= 0.1:
            r -= 20.0                               # смерть
        if dead and not self.prev_dead:
            pydirectinput.press('r')
            r -= 30.0                               # чётко умер

        style_gain = style - self.prev_style
        if style_gain > 0:
            r += style_gain * 0.1                   # бонус за стиль/убийства
            r += style_gain * 0.15                  # бонус за стиль/убийства
            if style_gain > 50:
                r += 1.0                            # испытания оружия
            if style_gain > 100:
                r += 5.0                            # много врагов
            if style_gain > 200:
                r += 10.0                           # убийство массы врагов
            self.frames_since_style = 0
        else:
            self.frames_since_style += 1

        if style_rank != self.prev_style_rank:
            r += STYLE_REWARD.get(style_rank, 0.0)
            self.prev_style_rank = style_rank

        if self.frames_since_style > 30 and (action[7] or action[8]):
            r -= 0.1                                # стрельба без врагов

        r += (rail - self.prev_rail) * 2.0          # заряд rail

        # bonus for active movement
        r += 0.05 * float(action[IDX_SPACE])
        r += 0.05 * float(action[IDX_SHIFT])
        r += 0.05 * float(action[IDX_CTRL])

        if action[IDX_SHIFT] and hp_loss == 0:
            r += 0.5                                # уворот без получения урона
        if dash > self.prev_dash and not action[IDX_SPACE]:
            r += 0.2                                # dash восстановлен
        if action[IDX_SHIFT] and dash < 0.1:
            r -= 0.3                                # спам dash

        if flash and not self.prev_flash:
            r += 5.0                                # парри

        # penalize keeping the view pitched up or down for long
        if self.frames_look_down > 60:
            r -= 0.05
        if self.frames_look_up > 60:
            r -= 0.05

        # --- смена оружия ---
        cur_slot = None
        variant_switch = False
        for i in range(5):                 # слоты 1-5
            pressed = action[SLOT_OFF+i] > 0
            if pressed and cur_slot is None:
                cur_slot = i
            if pressed and i == self.prev_slot and not self.prev_slot_pressed[i]:
                variant_switch = True
            self.prev_slot_pressed[i] = pressed
        if cur_slot is not None and cur_slot != self.prev_slot:
            r += 0.5                       # бонус за смену
            self.frames_since_slot = 0

        if cur_slot is not None:
            if variant_switch:
                r += 0.3                   # переключение варианта оружия
                self.frames_since_slot = 0
            elif cur_slot != self.prev_slot:
                r += 0.5                   # бонус за смену оружия
                self.frames_since_slot = 0
            self.prev_slot = cur_slot
        else:
            self.frames_since_slot += 1
            if self.frames_since_slot > 360:   # >6 сек без смены
                r -= 0.05

        # --- save prev ---
        self.prev_hp, self.prev_dash  = hp, dash
        self.prev_rail, self.prev_style = rail, style
        self.prev_style_rank = style_rank
        self.prev_flash = flash
        self.prev_dead = dead
        self.prev_frame = frame

        return obs, r, False, False, {}    # (no terminal flag yet)

    def render(self, *a, **kw):
        pass

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
# --------------------------------------------------------------------
#  Minimal Gym-обёртка, чтобы SB3 не ругался
# --------------------------------------------------------------------
import gym
class UltraKillWrapper(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, mouse=False, mouse_scale=30):
        super().__init__()
        self.core = UltraKillEnv(mouse=mouse, mouse_scale=mouse_scale)
        self.action_space      = self.core.action_space
        self.observation_space = self.core.observation_space
    def reset(self, seed=None, options=None):
        obs, info = self.core.reset()
        return obs, info
    def step(self, action):
        return self.core.step(action)
    def render(self, *a, **kw):
        return None
    def close(self):
        self.core.close()
