# player.py

import random
import pygame
from constants import *

class Player:
    """Абстрактный класс для всех типов игроков."""
    def __init__(self, player_id):
        self.player_id = player_id

    def get_move(self, game_state, events):
        """Возвращает ход, сделанный игроком. Должен быть переопределен."""
        raise NotImplementedError

class BotPlayer(Player):
    """Игрок, управляемый простым случайным AI."""
    def get_move(self, game_state, events):
        moves = game_state.get_legal_moves()
        if len(moves) == 1 and moves[0] == ('end_turn',):
            return moves[0]
        else:
            non_end_moves = [m for m in moves if m != ('end_turn',)]
            if non_end_moves and random.random() > 0.1:
                return random.choice(non_end_moves)
            else:
                return ('end_turn',)

class HumanPlayer(Player):
    """Игрок, управляемый человеком через UI."""
    def __init__(self, player_id):
        super().__init__(player_id)
        self.selected_unit_pos = None

    def get_move(self, game_state, events, ui_elements):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if ui_elements['end_turn_btn'].collidepoint(mouse_pos):
                    self.selected_unit_pos = None
                    return ('end_turn',)
                
                y_offset = ui_elements['board_y_offset']
                cell_size = ui_elements['cell_size']
                if mouse_pos[1] < y_offset: continue
                
                c = mouse_pos[0] // cell_size
                r = (mouse_pos[1] - y_offset) // cell_size

                if not (0 <= r < game_state.board_size and 0 <= c < game_state.board_size):
                    self.selected_unit_pos = None; continue

                clicked_pos = (r, c)
                legal_moves = game_state.get_legal_moves()

                ### НОВАЯ ЛОГИКА УПРАВЛЕНИЯ ###

                # 1. Левая кнопка мыши: Выбор, перемещение, слияние
                if event.button == 1:
                    if self.selected_unit_pos:
                        from_pos = self.selected_unit_pos
                        move_action = ('move', from_pos, clicked_pos)
                        merge_action = ('merge', tuple(sorted((from_pos, clicked_pos))))
                        
                        if move_action in legal_moves:
                            self.selected_unit_pos = None; return move_action
                        elif merge_action in legal_moves:
                            self.selected_unit_pos = None; return merge_action
                        else:
                            unit_at_click = game_state.units_board[clicked_pos] * self.player_id
                            if unit_at_click > 0 and clicked_pos not in game_state.moved_units_this_turn:
                                self.selected_unit_pos = clicked_pos
                            else:
                                self.selected_unit_pos = None
                    else: # Юнит не выбран
                        unit_at_click = game_state.units_board[clicked_pos]
                        if unit_at_click * self.player_id > 0 and clicked_pos not in game_state.moved_units_this_turn:
                            self.selected_unit_pos = clicked_pos
                
                # 2. Колесико мыши: Постройка воина
                elif event.button == 2:
                    self.selected_unit_pos = None
                    build_action = ('build_warrior', clicked_pos)
                    if build_action in legal_moves:
                        return build_action

                # 3. Правая кнопка мыши: Постройка крестьянина
                elif event.button == 3:
                    self.selected_unit_pos = None
                    build_action = ('build_peasant', clicked_pos)
                    if build_action in legal_moves:
                        return build_action
                        
        return None
