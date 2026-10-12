"""Draws a Game with pygame. Read-only: rendering never changes game state.

Shapes only for now. Swap in open-source sprites later without touching the rules.
"""

from __future__ import annotations

import math

import pygame

from mazehunt.config import GameConfig
from mazehunt.core.actions import DELTAS, Pos
from mazehunt.core.game import Game
from mazehunt.core.maze import Maze

HUD_HEIGHT = 48
COLORS = {
    "bg": (14, 15, 20),
    "floor": (30, 32, 42),
    "floor_dot": (44, 47, 60),
    "wall": (64, 72, 96),
    "wall_edge": (92, 102, 134),
    "exit": (58, 170, 92),
    "exit_inner": (30, 96, 52),
    "player": (86, 204, 255),
    "gun": (220, 240, 255),
    "bullet": (255, 216, 90),
    "text": (232, 232, 240),
    "dim": (140, 142, 162),
    "ammo_empty": (70, 72, 88),
    "good": (110, 220, 130),
    "bad": (245, 96, 96),
}
MONSTER_COLORS = [(236, 82, 82), (242, 152, 60), (188, 96, 224), (92, 204, 144)]


class Renderer:
    def __init__(self, maze: Maze, config: GameConfig, tile: int = 32):
        self.maze, self.config, self.tile = maze, config, tile
        self.size = (maze.width * tile, maze.height * tile + HUD_HEIGHT)
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_big = pygame.font.SysFont("consolas", 40, bold=True)
        self.background = self._build_background()

    def _build_background(self) -> pygame.Surface:
        t = self.tile
        surf = pygame.Surface((self.maze.width * t, self.maze.height * t))
        surf.fill(COLORS["floor"])
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                rect = pygame.Rect(x * t, y * t, t, t)
                if self.maze.is_wall((x, y)):
                    pygame.draw.rect(surf, COLORS["wall"], rect)
                    pygame.draw.rect(surf, COLORS["wall_edge"], rect, width=1)
                else:
                    pygame.draw.circle(surf, COLORS["floor_dot"], rect.center, 1)
        ex, ey = self.maze.exit
        door = pygame.Rect(ex * t, ey * t, t, t).inflate(-6, -6)
        pygame.draw.rect(surf, COLORS["exit"], door, border_radius=4)
        pygame.draw.rect(surf, COLORS["exit_inner"], door.inflate(-10, -10), border_radius=3)
        return surf

    def _center(self, prev: Pos, pos: Pos, alpha: float) -> tuple[float, float]:
        """Pixel center of an entity part-way (alpha 0→1) between last tick and this tick."""
        t = self.tile
        x = prev[0] + (pos[0] - prev[0]) * alpha
        y = prev[1] + (pos[1] - prev[1]) * alpha
        return (x * t + t / 2, y * t + t / 2 + HUD_HEIGHT)

    def draw(self, screen: pygame.Surface, game: Game, alpha: float,
             level: int, lives: int, monster_label: str) -> None:
        screen.fill(COLORS["bg"])
        screen.blit(self.background, (0, HUD_HEIGHT))
        t = self.tile
        for b in game.bullets:
            pygame.draw.circle(screen, COLORS["bullet"], self._center(b.prev_pos, b.pos, alpha), t * 0.14)
        for m in game.monsters:
            if m.alive:
                self._draw_monster(screen, m, alpha)
                if game.tick - m.last_hit_tick <= 1:
                    pygame.draw.circle(screen, (255, 255, 255), self._center(m.prev_pos, m.pos, alpha), t * 0.46, 3)
            elif game.tick - m.last_hit_tick <= 3:
                # Death burst: a ring that grows and fades over 3 ticks.
                age = game.tick - m.last_hit_tick + alpha
                color = MONSTER_COLORS[m.index % len(MONSTER_COLORS)]
                pygame.draw.circle(screen, color, self._center(m.pos, m.pos, 0), t * (0.3 + 0.25 * age),
                                   max(1, int(5 - age)))
        self._draw_player(screen, game, alpha)
        self._draw_hud(screen, game, level, lives, monster_label)

    def _draw_monster(self, screen: pygame.Surface, m, alpha: float) -> None:
        t = self.tile
        cx, cy = self._center(m.prev_pos, m.pos, alpha)
        pygame.draw.circle(screen, MONSTER_COLORS[m.index % len(MONSTER_COLORS)], (cx, cy), t * 0.4)
        dx, dy = DELTAS[m.facing]
        for side in (-1, 1):
            # Two eyes side by side, shifted toward where the monster is heading.
            ex = cx + side * t * 0.15 + dx * t * 0.1
            ey = cy - t * 0.08 + dy * t * 0.1
            pygame.draw.circle(screen, (255, 255, 255), (ex, ey), t * 0.11)
            pygame.draw.circle(screen, (20, 20, 20), (ex + dx * t * 0.04, ey + dy * t * 0.04), t * 0.05)
        pip_w, gap = t * 0.18, t * 0.05
        total = self.config.monster_hp * pip_w + (self.config.monster_hp - 1) * gap
        for i in range(self.config.monster_hp):
            color = COLORS["bad"] if i < m.hp else COLORS["ammo_empty"]
            rect = pygame.Rect(cx - total / 2 + i * (pip_w + gap), cy - t * 0.62, pip_w, t * 0.1)
            pygame.draw.rect(screen, color, rect)

    def _draw_player(self, screen: pygame.Surface, game: Game, alpha: float) -> None:
        t, p = self.tile, game.player
        cx, cy = self._center(p.prev_pos, p.pos, alpha)
        dx, dy = DELTAS[p.facing]
        pygame.draw.line(screen, COLORS["gun"], (cx, cy), (cx + dx * t * 0.48, cy + dy * t * 0.48), 5)
        pygame.draw.circle(screen, COLORS["player"], (cx, cy), t * 0.34)

    def _draw_hud(self, screen: pygame.Surface, game: Game, level: int, lives: int, monster_label: str) -> None:
        seconds = math.ceil(game.ticks_left / self.config.ticks_per_second)
        text = self.font.render(f"LEVEL {level}   LIVES {lives}   TIME {seconds:>2}s   AMMO", True, COLORS["text"])
        screen.blit(text, (12, (HUD_HEIGHT - text.get_height()) // 2))
        x = 12 + text.get_width() + 14
        for i in range(self.config.max_ammo):
            color = COLORS["bullet"] if i < game.player.ammo else COLORS["ammo_empty"]
            pygame.draw.circle(screen, color, (x + i * 18, HUD_HEIGHT // 2), 6)
        label = self.font.render(f"monsters: {monster_label}", True, COLORS["dim"])
        screen.blit(label, (self.size[0] - label.get_width() - 12, (HUD_HEIGHT - label.get_height()) // 2))

    def draw_overlay(self, screen: pygame.Surface, title: str, lines: list[str],
                     title_color: tuple[int, int, int] | None = None) -> None:
        shade = pygame.Surface(self.size, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 175))
        screen.blit(shade, (0, 0))
        head = self.font_big.render(title, True, title_color or COLORS["text"])
        y = self.size[1] * 0.36
        screen.blit(head, ((self.size[0] - head.get_width()) // 2, y))
        y += head.get_height() + 18
        for line in lines:
            surf = self.font.render(line, True, COLORS["text"] if line else COLORS["dim"])
            screen.blit(surf, ((self.size[0] - surf.get_width()) // 2, y))
            y += 28
