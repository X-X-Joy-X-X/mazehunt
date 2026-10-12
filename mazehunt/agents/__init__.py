"""Monster controllers.

Every controller has the same interface, so the game can't tell a scripted monster from a
learned one:

    actions = controller.act(game)   # one action per monster, in game.monsters order

Week 1: scripted baselines (scripted_monsters.py).
Week 2+: MAPPO / IPPO learned teams, and the ghost player (behavior cloning).
"""
