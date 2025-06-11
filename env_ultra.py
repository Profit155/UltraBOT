# env_ultra.py — RL-обёртка ULTRAKILL V1

try:
    import gym                           # старый OpenAI Gym
except ImportError:
    import gymnasium as gym              # если gym нет, берём gymnasium

from gym import spaces
import numpy as np, cv2, mss, time, pydirectinput
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

# ───────────
class UltraKillEnv:
    def __init__(self, res=(84,84), mouse=False, mouse_scale=30):
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

        self.observation_space = spaces.Box(0,255,shape=(3,*res),dtype=np.uint8)
        if mouse:
            self.action_space = spaces.Box(-1.0, 1.0,
                                          shape=(len(A_KEYS)+2,),
                                          dtype=np.float32)
        else:
            self.action_space = spaces.MultiBinary(len(A_KEYS))

        self.sct = mss.mss();  self.mon = self.sct.monitors[1]

        # prev-состояние для reward
        self._reset_prev()
        self.frames_since_slot = 0
        self.prev_slot         = None

    # ---------------- utils ----------------
    def _grab(self):
        scr = np.asarray(self.sct.grab(self.mon))[:,:,:3]         # BGR
        return cv2.resize(scr, self.res, interpolation=cv2.INTER_AREA)

    def _reset_prev(self):
        self.prev_hp = 1.0
        self.prev_dash = self.prev_rail = 1.0
        self.prev_style = 0
        self.prev_flash = False
        self.prev_dead = False
        self.frames_look_down = 0
        self.frames_look_up   = 0

    # ---------------- Gym API --------------
    def reset(self, *, seed=None, options=None):
        self._reset_prev()
        self.frames_since_slot = 0
        self.prev_slot = None
        return self._grab().transpose(2,0,1), {}

    def step(self, action):
        """Send actions and compute reward.

        In addition to the usual damage/style rewards, this function
        discourages keeping the camera pointed straight up or down for
        extended periods by inspecting frame brightness.
        """
        # --- клавиши и мышь ---
        dx = dy = 0
        if self.mouse and len(action) >= len(A_KEYS)+2:
            dx = int(float(action[len(A_KEYS)])   * self.mouse_scale)
            dy = int(float(action[len(A_KEYS)+1]) * self.mouse_scale)
            # clip excessive movement to avoid wild spins
            mx = int(self.mouse_scale * 0.75)
            dx = int(np.clip(dx, -mx, mx))
            dy = int(np.clip(dy, -mx, mx))

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
        frame = self._grab()                           # BGR 84×84×3
        obs   = frame.transpose(2,0,1)
        hp    = frame[76:80,  2:16, 2].mean()/255      # красная полоска
        dash  = frame[80:82,  2:16, 0].mean()/255      # голубой dash
        rail  = frame[82:83,  2:16, 1].mean()/255      # бирюза rail
        style = (frame[14:23, 67:83, :] > 200).sum()   # белые буквы
        flash = frame.mean() > 240
        dark  = frame.mean() < 30
        words = (frame[30:60, 20:64, :] > 200).mean() > 0.02
        dead  = dark and words

        # --- vertical look estimate via brightness profile ---
        # bright bottom rows usually mean the camera faces downward and
        # bright top rows the opposite. Counters accumulate how long
        # the agent stares down or up to apply a small penalty later.
        vert_profile = frame.mean(axis=2).mean(axis=1)
        top_b  = vert_profile[:20].mean()
        bot_b  = vert_profile[-20:].mean()
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
            if style_gain > 100:
                r += 5.0                            # много врагов
            if style_gain > 200:
                r += 10.0                           # убийство массы врагов

        r += (rail - self.prev_rail) * 2.0          # заряд rail

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
        for i in range(5):                 # слоты 1-5
            if action[SLOT_OFF+i]:
                cur_slot = i
        if cur_slot is not None and cur_slot != self.prev_slot:
            r += 0.5                       # бонус за смену
            self.frames_since_slot = 0
            self.prev_slot = cur_slot
        else:
            self.frames_since_slot += 1
            if self.frames_since_slot > 360:   # >6 сек без смены
                r -= 0.05

        # --- save prev ---
        self.prev_hp, self.prev_dash  = hp, dash
        self.prev_rail, self.prev_style = rail, style
        self.prev_flash = flash
        self.prev_dead = dead

        return obs, r, False, False, {}    # (no terminal flag yet)

    def render(self, *a, **kw): pass
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
        pass



