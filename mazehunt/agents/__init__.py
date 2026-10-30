"""Monster controllers.

Every controller has the same interface, so the game can't tell a scripted monster from a
learned one:

    actions = controller.act(game)   # one action per monster, in game.monsters order

scripted_monsters.py  hand-written baselines (greedy chasers, random)
mappo.py              MAPPO / IPPO learned teams (LearnedMonsters) and the PPO trainer
The ghost player (behavior cloning of the human) lives in mazehunt/adapt/ghost.py.
"""
