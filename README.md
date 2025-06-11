# UltraBOT
нейросеть которая играет в ULTRAKILL

## Exploration rewards

The environment now tracks frame-to-frame changes to discourage camping.
If the scene does not change for about three seconds a small penalty is
given. Significant visual changes provide a bonus and reset the timer. In
addition, hitting a game checkpoint ("CHECKPOINT" text on screen) grants a
one-time reward.
