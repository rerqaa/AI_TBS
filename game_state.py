# game_state.py

import numpy as np
import copy
from constants import *
import numba # <<< НОВОЕ: Импортируем Numba

# <<< НОВОЕ: Создаем jitclass для констант, чтобы Numba мог их использовать в скомпилированном коде >>>
spec = [
    ('PLAYER_1', numba.int8), ('PLAYER_2', numba.int8), ('NEUTRAL', numba.int8),
    ('EMPTY', numba.int8), ('PEASANT_P1', numba.int8), ('WARRIOR_P1', numba.int8),
    ('PEASANT_P2', numba.int8), ('WARRIOR_P2', numba.int8),
    ('PEASANT_COST', numba.int8), ('PEASANT_UPKEEP', numba.int8),
    ('WARRIOR_COST', numba.int8), ('WARRIOR_MERGE_COST', numba.int8),
    ('WARRIOR_UPKEEP', numba.int8), ('INCOME_PER_CELL', numba.int8),
    ('BOARD_SIZE', numba.int8), ('MAX_TURNS', numba.int8),
]
@numba.experimental.jitclass(spec)
class NumbaConstants:
    def __init__(self):
        self.PLAYER_1 = PLAYER_1; self.PLAYER_2 = PLAYER_2; self.NEUTRAL = NEUTRAL
        self.EMPTY = EMPTY; self.PEASANT_P1 = PEASANT_P1; self.WARRIOR_P1 = WARRIOR_P1
        self.PEASANT_P2 = PEASANT_P2; self.WARRIOR_P2 = WARRIOR_P2
        self.PEASANT_COST = PEASANT_COST; self.PEASANT_UPKEEP = PEASANT_UPKEEP
        self.WARRIOR_COST = WARRIOR_COST; self.WARRIOR_MERGE_COST = WARRIOR_MERGE_COST
        self.WARRIOR_UPKEEP = WARRIOR_UPKEEP; self.INCOME_PER_CELL = INCOME_PER_CELL
        self.BOARD_SIZE = BOARD_SIZE; self.MAX_TURNS = MAX_TURNS

C = NumbaConstants()


class GameState:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size; self.turn_count = 0; self.current_player = PLAYER_1
        self.units_board = np.zeros((board_size, board_size), dtype=np.int8)
        self.territory_board = np.zeros((board_size, board_size), dtype=np.int8)
        self.player_coins = {PLAYER_1: 0, PLAYER_2: 0}
        # <<< ИЗМЕНЕНИЕ: Заменяем set на NumPy массив для совместимости с Numba >>>
        self.moved_units_this_turn = np.zeros((board_size, board_size), dtype=np.bool_)
        self._setup_initial_state()

    def _setup_initial_state(self):
        self.units_board[0, 0] = PEASANT_P1; self.territory_board[0, 0] = PLAYER_1
        p2_pos = (self.board_size - 1, self.board_size - 1)
        self.units_board[p2_pos] = PEASANT_P2; self.territory_board[p2_pos] = PLAYER_2

    def clone(self):
        new_state = GameState(self.board_size)
        new_state.turn_count = self.turn_count
        new_state.current_player = self.current_player
        new_state.units_board = np.copy(self.units_board)
        new_state.territory_board = np.copy(self.territory_board)
        new_state.player_coins = self.player_coins.copy()
        # <<< ИЗМЕНЕНИЕ: Копируем NumPy массив >>>
        new_state.moved_units_this_turn = np.copy(self.moved_units_this_turn)
        return new_state

    def get_legal_moves(self):
        # <<< ИЗМЕНЕНИЕ: Эта функция теперь является оберткой для быстрой Numba-версии >>>
        my_coins = self.player_coins[self.current_player]
        
        raw_moves = _get_legal_moves_numba(
            self.units_board, self.territory_board, self.moved_units_this_turn,
            self.current_player, my_coins, C
        )
        
        legal_moves = []
        for move_data in raw_moves:
            move_type = move_data[0]
            if move_type == 1: # move
                legal_moves.append(('move', (move_data[1], move_data[2]), (move_data[3], move_data[4])))
            elif move_type == 2: # merge
                pos1 = (move_data[1], move_data[2]); pos2 = (move_data[3], move_data[4])
                legal_moves.append(('merge', tuple(sorted((pos1, pos2)))))
            elif move_type == 3: # build_peasant
                legal_moves.append(('build_peasant', (move_data[1], move_data[2])))
            elif move_type == 4: # build_warrior
                legal_moves.append(('build_warrior', (move_data[1], move_data[2])))
        
        legal_moves.append(('end_turn',))
        return sorted(list(set(legal_moves)))

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
            new_state.moved_units_this_turn[to_pos] = True # <<< ИЗМЕНЕНИЕ
        elif move_type == 'build_peasant':
            pos = move[1]
            new_state.player_coins[new_state.current_player] -= PEASANT_COST
            new_state.units_board[pos] = my_peasant_type
            new_state.moved_units_this_turn[pos] = True # <<< ИЗМЕНЕНИЕ
        elif move_type == 'build_warrior':
            pos = move[1]
            new_state.player_coins[new_state.current_player] -= WARRIOR_COST
            new_state.units_board[pos] = my_warrior_type
            new_state.moved_units_this_turn[pos] = True # <<< ИЗМЕНЕНИЕ
        elif move_type == 'merge':
            pos1, pos2 = move[1][0], move[1][1]
            new_state.player_coins[new_state.current_player] -= WARRIOR_MERGE_COST
            new_state.units_board[pos1] = EMPTY; new_state.units_board[pos2] = my_warrior_type
            new_state.moved_units_this_turn[pos2] = True # <<< ИЗМЕНЕНИЕ
        elif move_type == 'end_turn':
            new_state._end_turn()
        return new_state

    def _end_turn(self):
        player_ending_turn = self.current_player
        income = np.sum(self.territory_board == player_ending_turn) * INCOME_PER_CELL
        num_peasants = np.sum(self.units_board * player_ending_turn == 1)
        num_warriors = np.sum(self.units_board * player_ending_turn == 2)
        upkeep = (num_peasants * PEASANT_UPKEEP) + (num_warriors * WARRIOR_UPKEEP)
        self.player_coins[player_ending_turn] += (income - upkeep)
        if self.player_coins[player_ending_turn] < 0:
            self.player_coins[player_ending_turn] = 0
        if player_ending_turn == PLAYER_2:
            self.turn_count += 1
        self.moved_units_this_turn.fill(False) # <<< ИЗМЕНЕНИЕ
        self.current_player *= -1

    def get_game_ended(self, max_turns=MAX_TURNS):
        # <<< ИЗМЕНЕНИЕ: Вызываем быструю Numba-версию >>>
        return _get_game_ended_numba(
            self.units_board, self.territory_board, self.current_player,
            self.turn_count, max_turns, self.board_size
        )

# <<< НОВАЯ, СКОМПИЛИРОВАННАЯ ФУНКЦИЯ для get_legal_moves >>>
@numba.njit(cache=True)
def _get_legal_moves_numba(units_board, territory_board, moved_units_this_turn, current_player, my_coins, C):
    legal_moves = numba.typed.List()
    board_size = C.BOARD_SIZE
    my_peasant_type = C.PEASANT_P1 if current_player == C.PLAYER_1 else C.PEASANT_P2
    my_warrior_type = C.WARRIOR_P1 if current_player == C.PLAYER_1 else C.WARRIOR_P2

    for r in range(board_size):
        for c in range(board_size):
            if units_board[r, c] * current_player > 0:
                if moved_units_this_turn[r, c]: continue
                unit_type = units_board[r, c]
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < board_size and 0 <= nc < board_size:
                            target_cell_unit = units_board[nr, nc]
                            can_move = False
                            if unit_type == my_warrior_type:
                                if target_cell_unit * current_player <= 0: can_move = True
                            elif unit_type == my_peasant_type:
                                if target_cell_unit == C.EMPTY: can_move = True
                            if can_move:
                                legal_moves.append(np.array([1, r, c, nr, nc], dtype=np.int16))
                            if unit_type == my_peasant_type and target_cell_unit == my_peasant_type and my_coins >= C.WARRIOR_MERGE_COST:
                                if r < nr or (r == nr and c < nc):
                                    legal_moves.append(np.array([2, r, c, nr, nc], dtype=np.int16))
    for r in range(board_size):
        for c in range(board_size):
            if territory_board[r, c] == current_player:
                if units_board[r, c] == C.EMPTY:
                    if my_coins >= C.PEASANT_COST:
                        legal_moves.append(np.array([3, r, c, 0, 0], dtype=np.int16))
                    if my_coins >= C.WARRIOR_COST:
                        legal_moves.append(np.array([4, r, c, 0, 0], dtype=np.int16))
    return legal_moves

# <<< НОВАЯ, СКОМПИЛИРОВАННАЯ ФУНКЦИЯ для get_game_ended >>>
@numba.njit(cache=True)
def _get_game_ended_numba(units_board, territory_board, current_player, turn_count, max_turns, board_size):
    my_units = 0
    for r in range(board_size):
        for c in range(board_size):
            if units_board[r, c] * current_player > 0:
                my_units += 1
    if my_units == 0:
        return -current_player

    p1_territory = 0; p2_territory = 0
    for r in range(board_size):
        for c in range(board_size):
            if territory_board[r, c] == 1: p1_territory += 1
            elif territory_board[r, c] == -1: p2_territory += 1

    if p1_territory == board_size ** 2: return 1
    if p2_territory == board_size ** 2: return -1
    
    if turn_count >= max_turns:
        if p1_territory > p2_territory: return 1
        if p2_territory > p1_territory: return -1
        return 1e-4
        
    return 0
