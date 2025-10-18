from constants import *
import numpy as np

def get_action_space():
    actions = []
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        actions.append(('move', (r, c), (nr, nc)))

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    action = ('merge', tuple(sorted(((r, c), (nr, nc)))))
                    if action not in actions:
                        actions.append(action)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            actions.append(('build_peasant', (r, c)))
            actions.append(('build_warrior', (r, c)))

    actions.append(('end_turn',))
    
    action_to_index = {action: i for i, action in enumerate(actions)}
    index_to_action = {i: action for i, action in enumerate(actions)}
    
    return action_to_index, index_to_action

ACTION_TO_INDEX, INDEX_TO_ACTION = get_action_space()
ACTION_SPACE_SIZE = len(ACTION_TO_INDEX)


MOVE_DIRECTIONS = {
    (-1, 0): 0,
    (-1, 1): 1,
    (0, 1): 2,
    (1, 1): 3,
    (1, 0): 4,
    (1, -1): 5,
    (0, -1): 6,
    (-1, -1): 7,
}

MERGE_DIRECTIONS = {
    (0, 1): 8,
    (1, 0): 9,
    (1, 1): 10,
    (1, -1): 11,
}

BUILD_PEASANT_PLANE = 12
BUILD_WARRIOR_PLANE = 13
END_TURN_PLANE = 14

NUM_ACTION_PLANES = 15

# --- НОВЫЙ КОД ДЛЯ АУГМЕНТАЦИИ ---

def create_symmetry_lookups(action_to_idx, idx_to_action, board_size):
    """
    Создает таблицы подстановки для 8 симметрий доски.
    Возвращает список из 8 numpy-массивов.
    Каждый массив `lookup` имеет размер ACTION_SPACE_SIZE.
    `lookup[old_action_idx]` дает `new_action_idx` после применения симметрии.
    """
    
    # Функции для трансформации координат
    def identity(r, c): return r, c
    def rot90(r, c): return c, board_size - 1 - r
    def rot180(r, c): return board_size - 1 - r, board_size - 1 - c
    def rot270(r, c): return board_size - 1 - c, r
    def flip_hr(r, c): return r, board_size - 1 - c # Горизонтальное отражение
    
    symmetries = [
        identity, rot90, rot180, rot270,
        lambda r, c: flip_hr(*identity(r, c)),
        lambda r, c: flip_hr(*rot90(r, c)),
        lambda r, c: flip_hr(*rot180(r, c)),
        lambda r, c: flip_hr(*rot270(r, c)),
    ]

    lookups = []
    for sym_func in symmetries:
        lookup = np.zeros(len(action_to_idx), dtype=np.int32)
        for i in range(len(idx_to_action)):
            action = idx_to_action[i]
            action_type = action[0]
            
            if action_type == 'move':
                r1, c1 = sym_func(*action[1])
                r2, c2 = sym_func(*action[2])
                new_action = ('move', (r1, c1), (r2, c2))
            elif action_type == 'merge':
                p1 = sym_func(*action[1][0])
                p2 = sym_func(*action[1][1])
                new_action = ('merge', tuple(sorted((p1, p2))))
            elif action_type.startswith('build'):
                r, c = sym_func(*action[1])
                new_action = (action_type, (r, c))
            elif action_type == 'end_turn':
                new_action = action
            
            lookup[i] = action_to_idx[new_action]
        lookups.append(lookup)
        
    return lookups

SYMMETRY_LOOKUPS = create_symmetry_lookups(ACTION_TO_INDEX, INDEX_TO_ACTION, BOARD_SIZE)
