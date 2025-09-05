# main.py

import pygame
import sys
from constants import *
from game_state import GameState
from visualizer import GameVisualizer
from player import BotPlayer, HumanPlayer

class GameApp:
    def __init__(self):
        self.visualizer = GameVisualizer()
        self.clock = pygame.time.Clock()
        self.game_mode = 'MENU' # 'MENU', 'IN_GAME'
        self.game_state = None
        self.players = {}
        self.active_player = None
        self.menu_buttons = {
            'bvb': {'rect': pygame.Rect(self.visualizer.WIDTH / 2 - 150, self.visualizer.HEIGHT / 2 - 60, 300, 50), 'text': 'Бот против Бота'},
            'pvb': {'rect': pygame.Rect(self.visualizer.WIDTH / 2 - 150, self.visualizer.HEIGHT / 2, 300, 50), 'text': 'Человек против Бота'},
            'pvp': {'rect': pygame.Rect(self.visualizer.WIDTH / 2 - 150, self.visualizer.HEIGHT / 2 + 60, 300, 50), 'text': 'Человек против Человека'}
        }

    def setup_game(self, mode):
        self.game_state = GameState()
        if mode == 'bvb':
            self.players = {PLAYER_1: BotPlayer(PLAYER_1), PLAYER_2: BotPlayer(PLAYER_2)}
        elif mode == 'pvb':
            self.players = {PLAYER_1: HumanPlayer(PLAYER_1), PLAYER_2: BotPlayer(PLAYER_2)}
        elif mode == 'pvp':
            self.players = {PLAYER_1: HumanPlayer(PLAYER_1), PLAYER_2: HumanPlayer(PLAYER_2)}
        self.active_player = self.players[self.game_state.current_player]
        self.game_mode = 'IN_GAME'

    def run(self):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if self.game_mode == 'MENU':
                self.run_menu(events)
            elif self.game_mode == 'IN_GAME':
                self.run_game(events)

            self.clock.tick(30) # Ограничиваем FPS

    def run_menu(self, events):
        self.visualizer.draw_menu(self.menu_buttons)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if self.menu_buttons['bvb']['rect'].collidepoint(mouse_pos): self.setup_game('bvb')
                elif self.menu_buttons['pvb']['rect'].collidepoint(mouse_pos): self.setup_game('pvb')
                elif self.menu_buttons['pvp']['rect'].collidepoint(mouse_pos): self.setup_game('pvp')

    def run_game(self, events):
        # Проверка клика по кнопке "В меню"
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.visualizer.ui_elements['menu_btn'].collidepoint(event.pos):
                    self.game_mode = 'MENU'
                    return

        # Проверка на конец игры
        winner = self.game_state.get_game_ended()
        if winner != 0:
            self.visualizer.draw_game_state(self.game_state, self.active_player)
            self.visualizer.show_winner_screen(winner)
            self.game_mode = 'MENU'
            return

        # Получение хода от активного игрока
        if isinstance(self.active_player, HumanPlayer):
            move = self.active_player.get_move(self.game_state, events, self.visualizer.ui_elements)
        else: # BotPlayer
            move = self.active_player.get_move(self.game_state, events)
            pygame.time.wait(100) # Небольшая задержка для бота

        if move:
            previous_player_id = self.game_state.current_player
            self.game_state = self.game_state.apply_move(move)
            
            # Если игрок сменился, обновляем активного игрока
            if self.game_state.current_player != previous_player_id:
                self.active_player = self.players[self.game_state.current_player]

        self.visualizer.draw_game_state(self.game_state, self.active_player)

if __name__ == '__main__':
    app = GameApp()
    app.run()
