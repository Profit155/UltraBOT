# UltraBOT
нейросеть которая играет в ULTRAKILL

## Exploration rewards

The environment now tracks frame-to-frame changes to discourage camping.
If the scene does not change for about three seconds a small penalty is
given. Significant visual changes provide a bonus and reset the timer. In
addition, hitting a game checkpoint ("CHECKPOINT" text on screen) grants a
one-time reward.

Weapon slots can be pressed again to cycle their variations. Trying a new
variant grants a small bonus. Large jumps of the style meter give extra
points to encourage flashy play. Shooting when no style was gained for a
while incurs a small penalty so the bot does not fire blindly.
