# UltraBOT
нейросеть которая играет в ULTRAKILL

## Exploration rewards

The environment now uses a larger 1024×768 observation frame which gives the
agent a clearer view of the stage. Frame differences are analysed to reward
movement and penalise camping. If the scene barely changes for roughly three
seconds a penalty is applied; large changes provide a small reward and reset
the timer. Hitting a game checkpoint ("CHECKPOINT" text on screen) grants a
one‑time bonus. When the end of a level is reached the central rank display is
detected and additional reward or penalty is applied based on its brightness,
roughly corresponding to ranks P through C.

Weapon slots can be pressed again to cycle their variations. Trying a new
variant grants a small bonus. Large jumps of the style meter give extra
points to encourage flashy play and help detect combo kills. Shooting when no
style was gained for a while incurs a small penalty so the bot does not fire
blindly. Movement that noticeably changes the scene grants a tiny reward each
frame so the agent keeps advancing through the map.
The environment also estimates whether the camera is pointed
too far up or down by comparing the brightness of the top and bottom
rows of the screen. Keeping such an orientation for over a second results
in a small penalty.
