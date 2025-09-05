# game_state.py

import numpy as np
import copy
from constants import *

class GameState:

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size; self.turn_count = 0; self.current_player = PLAYER_1
        self.units_board = np.zeros((board_size, board_size), dtype=np.int8)
        self.territory_board = np.zeros((board_size, board_size), dtype=np.int8)
        self.player_coins = {PLAYER_1: 0, PLAYER_2: 0}; self.moved_units_this_turn = set()
        self._setup_initial_state()
    def _setup_initial_state(self):
        self.units_board[0, 0] = PEASANT_P1; self.territory_board[0, 0] = PLAYER_1
        p2_pos = (self.board_size - 1, self.board_size - 1)
        self.units_board[p2_pos] = PEASANT_P2; self.territory_board[p2_pos] = PLAYER_2
    def clone(self): return copy.deepcopy(self)

    def get_legal_moves(self):
        legal_moves = []
        my_peasant_type = PEASANT_P1 if self.current_player == PLAYER_1 else PEASANT_P2
        my_warrior_type = WARRIOR_P1 if self.current_player == PLAYER_1 else WARRIOR_P2
        my_coins = self.player_coins[self.current_player]
        
        # ... (Логика ходов юнитов остается без изменений) ...
        unit_positions = np.argwhere(self.units_board * self.current_player > 0)
        for r, c in unit_positions:
            if (r, c) in self.moved_units_this_turn: continue
            unit_type = self.units_board[r, c]
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        target_cell_unit = self.units_board[nr, nc]; can_move = False
                        if unit_type in (my_warrior_type,):
                            if target_cell_unit * self.current_player <= 0: can_move = True
                        elif unit_type in (my_peasant_type,):
                            if target_cell_unit == EMPTY: can_move = True
                        if can_move: legal_moves.append(('move', (r, c), (nr, nc)))
                        if unit_type == my_peasant_type and target_cell_unit == my_peasant_type and my_coins >= WARRIOR_MERGE_COST:
                            legal_moves.append(('merge', tuple(sorted(((r, c), (nr, nc))))))

        # --- Логика постройки юнитов ---
        my_territory_pos = np.argwhere(self.territory_board == self.current_player)
        for r, c in my_territory_pos:
            if self.units_board[r, c] == EMPTY:
                # Покупка крестьянина
                if my_coins >= PEASANT_COST:
                    legal_moves.append(('build_peasant', (r, c)))
                
                ### НОВОВВЕДЕНИЕ: Покупка воина ###
                if my_coins >= WARRIOR_COST:
                    legal_moves.append(('build_warrior', (r, c)))

        legal_moves.append(('end_turn',))
        # Убираем дубликаты, которые могли возникнуть при слиянии
        return list(set(legal_moves))

    def apply_move(self, move):
        new_state = self.clone()
        move_type = move[0]
        my_peasant_type = PEASANT_P1 if new_state.current_player == PLAYER_1 else PEASANT_P2
        my_warrior_type = WARRIOR_P1 if new_state.current_player == PLAYER_1 else WARRIOR_P2

        if move_type == 'move':
            from_pos, to_pos = move[1], move[2]
            unit = new_state.units_board[from_pos]
            new_state.units_board[to_pos] = unit; new_state.units_board[from_pos] = EMPTY
            new_state.territory_board[to_pos] = new_state.current_player
            new_state.moved_units_this_turn.add(to_pos)
        elif move_type == 'build_peasant':
            pos = move[1]
            new_state.player_coins[new_state.current_player] -= PEASANT_COST
            new_state.units_board[pos] = my_peasant_type
            new_state.moved_units_this_turn.add(pos)
        
        ### НОВОВВЕДЕНИЕ: Обработка постройки воина ###
        elif move_type == 'build_warrior':
            pos = move[1]
            new_state.player_coins[new_state.current_player] -= WARRIOR_COST
            new_state.units_board[pos] = my_warrior_type
            new_state.moved_units_this_turn.add(pos)

        elif move_type == 'merge':
            pos1, pos2 = move[1][0], move[1][1]
            new_state.player_coins[new_state.current_player] -= WARRIOR_MERGE_COST
            new_state.units_board[pos1] = EMPTY; new_state.units_board[pos2] = my_warrior_type
            new_state.moved_units_this_turn.add(pos2)
        elif move_type == 'end_turn':
            new_state._end_turn()
        return new_state

    def _end_turn(self):
        """
        Внутренний метод для обработки конца хода игрока.
        Экономика и счетчик ходов обновляются только после хода второго игрока.
        """
        is_round_over = (self.current_player == PLAYER_2)

        # Если раунд завершен (походил второй игрок), обновляем экономику
        if is_round_over:
            # 1. Расчет дохода и расходов для обоих игроков
            for player in [PLAYER_1, PLAYER_2]:
                income = np.sum(self.territory_board == player) * INCOME_PER_CELL
                
                num_peasants = np.sum(self.units_board * player == 1)
                num_warriors = np.sum(self.units_board * player == 2)
                upkeep = (num_peasants * PEASANT_UPKEEP) + (num_warriors * WARRIOR_UPKEEP)
                
                self.player_coins[player] += (income - upkeep)
                if self.player_coins[player] < 0:
                    self.player_coins[player] = 0
            
            # Увеличиваем счетчик "глобальных" ходов
            self.turn_count += 1

        # Эти действия происходят после хода каждого игрока:
        # Очищаем список походивших юнитов для следующего игрока
        self.moved_units_this_turn.clear()

        # Смена игрока
        self.current_player *= -1

    def get_game_ended(self, max_turns=MAX_TURNS):
        p1_territory = np.sum(self.territory_board == PLAYER_1)
        p2_territory = np.sum(self.territory_board == PLAYER_2)
        if p1_territory == self.board_size ** 2: return PLAYER_1
        if p2_territory == self.board_size ** 2: return PLAYER_2
        p1_units = np.sum(self.units_board * PLAYER_1 > 0)
        p2_units = np.sum(self.units_board * PLAYER_2 > 0)
        if p2_units == 0 and self.player_coins[PLAYER_2] < PEASANT_COST: return PLAYER_1
        if p1_units == 0 and self.player_coins[PLAYER_1] < PEASANT_COST: return PLAYER_2
        if self.turn_count >= max_turns:
            if p1_territory > p2_territory: return PLAYER_1
            if p2_territory > p1_territory: return PLAYER_2
            return 1e-4
        return 0
