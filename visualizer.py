# visualizer.py

import pygame
from constants import *
from player import HumanPlayer 

class GameVisualizer:
    def __init__(self):
        pygame.init()
        self.CELL_SIZE = 60; self.INFO_PANEL_HEIGHT = 80
        self.WIDTH = BOARD_SIZE * self.CELL_SIZE
        self.HEIGHT = BOARD_SIZE * self.CELL_SIZE + self.INFO_PANEL_HEIGHT
        self.COLOR_P1 = (0, 120, 255); self.COLOR_P2 = (255, 50, 50)
        self.TERRITORY_P1 = (200, 225, 255); self.TERRITORY_P2 = (255, 200, 200)
        self.COLOR_GRID = (50, 50, 50); self.COLOR_BG = (20, 20, 20)
        self.COLOR_TEXT = (240, 240, 240); self.COLOR_BUTTON = (80, 80, 80)
        self.COLOR_BUTTON_HOVER = (110, 110, 110); self.COLOR_SELECTION = (255, 255, 0)
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Стратегия 10x10")
        self.font_info = pygame.font.SysFont("Arial", 24)
        self.font_menu = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_winner = pygame.font.SysFont("Arial", 48, bold=True)
        self.ui_elements = {
            'board_y_offset': self.INFO_PANEL_HEIGHT,
            'cell_size': self.CELL_SIZE
        }

    def draw_menu(self, buttons):
        self.screen.fill(self.COLOR_BG)
        title = self.font_menu.render("Стратегия 10x10", True, self.COLOR_TEXT)
        title_rect = title.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 4))
        self.screen.blit(title, title_rect)
        mouse_pos = pygame.mouse.get_pos()
        for name, data in buttons.items():
            rect = data['rect']; color = self.COLOR_BUTTON_HOVER if rect.collidepoint(mouse_pos) else self.COLOR_BUTTON
            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            text = self.font_info.render(data['text'], True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        pygame.display.flip()

    def draw_game_state(self, game_state, active_player=None):
        self.screen.fill(self.COLOR_BG)
        self._draw_info_panel(game_state)
        self._draw_board(game_state, active_player)
        self._draw_ingame_ui(active_player)
        pygame.display.flip()

    def _draw_info_panel(self, game_state):
        turn_text = self.font_info.render(f"Ход: {game_state.turn_count}", True, self.COLOR_TEXT)
        player_turn = "Игрок 1 (Синие)" if game_state.current_player == PLAYER_1 else "Игрок 2 (Красные)"
        player_color = self.COLOR_P1 if game_state.current_player == PLAYER_1 else self.COLOR_P2
        turn_queue_text = self.font_info.render(f"Очередь: {player_turn}", True, player_color)
        p1_coins = game_state.player_coins[PLAYER_1]; p2_coins = game_state.player_coins[PLAYER_2]
        coins_text_str = f"Монеты: {p1_coins} (С) / {p2_coins} (К)"; coins_text = self.font_info.render(coins_text_str, True, self.COLOR_TEXT)
        coins_rect = coins_text.get_rect(); coins_rect.topright = (self.WIDTH - 10, 40)
        self.screen.blit(turn_text, (10, 10)); self.screen.blit(turn_queue_text, (10, 40)); self.screen.blit(coins_text, coins_rect)

    def _draw_board(self, game_state, active_player):
        y_offset = self.INFO_PANEL_HEIGHT
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = pygame.Rect(c * self.CELL_SIZE, y_offset + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                territory_owner = game_state.territory_board[r, c]
                if territory_owner == PLAYER_1: pygame.draw.rect(self.screen, self.TERRITORY_P1, rect)
                elif territory_owner == PLAYER_2: pygame.draw.rect(self.screen, self.TERRITORY_P2, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                
                # <<< ИСПРАВЛЕНИЕ: Правильная проверка для NumPy массива >>>
                if game_state.moved_units_this_turn[r, c]:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill((0,0,0,80)); self.screen.blit(s, (rect.x, rect.y))
                
                if hasattr(active_player, 'selected_unit_pos') and active_player.selected_unit_pos == (r, c):
                    pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3)
                unit_type = game_state.units_board[r, c]
                self._draw_unit(unit_type, rect)

    def _draw_unit(self, unit_type, rect):
        center = rect.center; radius = self.CELL_SIZE // 3
        if unit_type == PEASANT_P1: pygame.draw.circle(self.screen, self.COLOR_P1, center, radius)
        elif unit_type == WARRIOR_P1:
            warrior_rect = pygame.Rect(0, 0, radius * 1.8, radius * 1.8); warrior_rect.center = center
            pygame.draw.rect(self.screen, self.COLOR_P1, warrior_rect)
        elif unit_type == PEASANT_P2: pygame.draw.circle(self.screen, self.COLOR_P2, center, radius)
        elif unit_type == WARRIOR_P2:
            warrior_rect = pygame.Rect(0, 0, radius * 1.8, radius * 1.8); warrior_rect.center = center
            pygame.draw.rect(self.screen, self.COLOR_P2, warrior_rect)
    
    def _draw_ingame_ui(self, active_player):
        menu_btn_rect = pygame.Rect(self.WIDTH - 130, 5, 120, 30)
        pygame.draw.rect(self.screen, self.COLOR_BUTTON, menu_btn_rect, border_radius=5)
        text = self.font_info.render("В меню", True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=menu_btn_rect.center)
        self.screen.blit(text, text_rect)
        self.ui_elements['menu_btn'] = menu_btn_rect
        
        if isinstance(active_player, HumanPlayer):
            end_turn_btn_rect = pygame.Rect(self.WIDTH // 2 - 100, self.HEIGHT - 45, 200, 40)
            self.ui_elements['end_turn_btn'] = end_turn_btn_rect
            pygame.draw.rect(self.screen, self.COLOR_BUTTON, end_turn_btn_rect, border_radius=5)
            text = self.font_info.render("Завершить ход", True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=end_turn_btn_rect.center)
            self.screen.blit(text, text_rect)

    def show_winner_screen(self, winner):
        if winner == 1e-4: text = "Ничья!"; color = self.COLOR_TEXT
        elif winner == PLAYER_1: text = "Победили Синие!"; color = self.COLOR_P1
        else: text = "Победили Красные!"; color = self.COLOR_P2
        winner_text = self.font_winner.render(text, True, color)
        text_rect = winner_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0)); self.screen.blit(winner_text, text_rect)
        pygame.display.flip()
        pygame.time.wait(2500)
