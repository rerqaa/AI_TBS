# main.py

import pygame
import sys
import torch
import os
from constants import *
from game_state import GameState
from visualizer import GameVisualizer
from model import StrategyNet
from player import HumanPlayer
from bots import RandomBotPlayer, AIPlayer

VISUAL_CPUCT = 1.5

class GameApp:
    def __init__(self):
        self.visualizer = GameVisualizer()
        self.clock = pygame.time.Clock()
        self.game_mode = 'MENU'
        self.game_state = None
        self.players = {}
        self.active_player = None
        self.menu_buttons = {
            'bvb': {'rect': pygame.Rect(self.visualizer.WIDTH / 2 - 150, self.visualizer.HEIGHT / 2 - 60, 300, 50), 'text': 'AI против AI'},
            'pvb': {'rect': pygame.Rect(self.visualizer.WIDTH / 2 - 150, self.visualizer.HEIGHT / 2, 300, 50), 'text': 'Человек против AI'},
            'pvp': {'rect': pygame.Rect(self.visualizer.WIDTH / 2 - 150, self.visualizer.HEIGHT / 2 + 60, 300, 50), 'text': 'Человек против Человека'}
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Игра запущена на устройстве: {self.device}")

        self.ai_model = StrategyNet().to(self.device)
        model_path = "best_model.pth"
        
        try:
            if os.path.exists(model_path):
                self.ai_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Обученная модель '{model_path}' успешно загружена.")
            else:
                print("Файл обученной модели не найден. AI будет играть со случайными весами.")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}. AI будет играть со случайными весами.")
        
        self.ai_model.eval()

    def setup_game(self, mode):
        self.game_state = GameState()
        if mode == 'bvb':
            # <<< ЭТО КЛЮЧЕВОЕ ИЗМЕНЕНИЕ №4: Передаем VISUAL_CPUCT в конструктор >>>
            self.players = {PLAYER_1: AIPlayer(PLAYER_1, self.ai_model, device=self.device, c_puct=VISUAL_CPUCT),
                            PLAYER_2: AIPlayer(PLAYER_2, self.ai_model, device=self.device, c_puct=VISUAL_CPUCT)}
        elif mode == 'pvb':
            self.players = {PLAYER_1: HumanPlayer(PLAYER_1), 
                            PLAYER_2: AIPlayer(PLAYER_2, self.ai_model, device=self.device, c_puct=VISUAL_CPUCT)}
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

            self.clock.tick(30)

    def run_menu(self, events):
        self.visualizer.draw_menu(self.menu_buttons)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if self.menu_buttons['bvb']['rect'].collidepoint(mouse_pos): self.setup_game('bvb')
                elif self.menu_buttons['pvb']['rect'].collidepoint(mouse_pos): self.setup_game('pvb')
                elif self.menu_buttons['pvp']['rect'].collidepoint(mouse_pos): self.setup_game('pvp')

    def run_game(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if 'menu_btn' in self.visualizer.ui_elements and self.visualizer.ui_elements['menu_btn'].collidepoint(event.pos):
                    self.game_mode = 'MENU'
                    return

        winner = self.game_state.get_game_ended()
        if winner != 0:
            self.visualizer.draw_game_state(self.game_state, self.active_player)
            self.visualizer.show_winner_screen(winner)
            self.game_mode = 'MENU'
            return

        move = None
        if isinstance(self.active_player, HumanPlayer):
            move = self.active_player.get_move(self.game_state, events, self.visualizer.ui_elements)
        elif isinstance(self.active_player, (AIPlayer, RandomBotPlayer)):
            move = self.active_player.get_move(self.game_state, events)

        if move:
            previous_player_id = self.game_state.current_player
            self.game_state = self.game_state.apply_move(move)
            
            if self.game_state.current_player != previous_player_id:
                self.active_player = self.players[self.game_state.current_player]

        self.visualizer.draw_game_state(self.game_state, self.active_player)

if __name__ == '__main__':
    app = GameApp()
    app.run()
