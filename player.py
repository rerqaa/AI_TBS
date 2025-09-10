# player.py

import pygame
from constants import *
from player_base import Player

class HumanPlayer(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.selected_unit_pos = None

    def get_move(self, game_state, events, ui_elements=None, *args, **kwargs):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                if 'end_turn_btn' in ui_elements and ui_elements['end_turn_btn'].collidepoint(mouse_pos):
                    self.selected_unit_pos = None
                    return ('end_turn',)
                
                y_offset = ui_elements['board_y_offset']
                cell_size = ui_elements['cell_size']
                if mouse_pos[1] < y_offset: continue
                
                c = mouse_pos[0] // cell_size
                r = (mouse_pos[1] - y_offset) // cell_size

                if not (0 <= r < game_state.board_size and 0 <= c < game_state.board_size):
                    self.selected_unit_pos = None
                    continue

                clicked_pos = (r, c)
                legal_moves = game_state.get_legal_moves()

                if event.button == 1: # Левая кнопка
                    if self.selected_unit_pos:
                        from_pos = self.selected_unit_pos
                        move_action = ('move', from_pos, clicked_pos)
                        merge_action = ('merge', tuple(sorted((from_pos, clicked_pos))))
                        
                        if move_action in legal_moves:
                            self.selected_unit_pos = None
                            return move_action
                        elif merge_action in legal_moves:
                            self.selected_unit_pos = None
                            return merge_action
                        else:
                            unit_at_click = game_state.units_board[clicked_pos] * self.player_id
                            # <<< ИСПРАВЛЕНИЕ: Правильная проверка для NumPy массива >>>
                            if unit_at_click > 0 and not game_state.moved_units_this_turn[clicked_pos]:
                                self.selected_unit_pos = clicked_pos
                            else:
                                self.selected_unit_pos = None
                    else: 
                        unit_at_click = game_state.units_board[clicked_pos]
                        # <<< ИСПРАВЛЕНИЕ: Правильная проверка для NumPy массива >>>
                        if unit_at_click * self.player_id > 0 and not game_state.moved_units_this_turn[clicked_pos]:
                            self.selected_unit_pos = clicked_pos
                
                elif event.button == 2: # Колесико
                    self.selected_unit_pos = None
                    build_action = ('build_warrior', clicked_pos)
                    if build_action in legal_moves:
                        return build_action

                elif event.button == 3: # Правая кнопка
                    self.selected_unit_pos = None
                    build_action = ('build_peasant', clicked_pos)
                    if build_action in legal_moves:
                        return build_action
                        
        return None
