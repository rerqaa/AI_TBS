# utils.py

from constants import *

def get_action_space():
    """
    Создает полное пространство всех возможных действий в игре и сопоставляет их с индексами.
    Возвращает:
        - action_to_index: словарь {'действие': индекс}
        - index_to_action: словарь {индекс: 'действие'}
    """
    actions = []
    
    # 1. Ходы перемещения (из каждой клетки в 8 направлениях)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        actions.append(('move', (r, c), (nr, nc)))

    # 2. Ходы слияния (из каждой клетки в 4 направлениях, чтобы избежать дублей)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # Только 'вправо' и 'вниз', чтобы ход (A,B) не дублировался с (B,A)
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    # Кортеж позиций отсортирован для консистентности
                    action = ('merge', tuple(sorted(((r, c), (nr, nc)))))
                    if action not in actions:
                        actions.append(action)

    # 3. Постройка юнитов (в каждой клетке)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            actions.append(('build_peasant', (r, c)))
            actions.append(('build_warrior', (r, c)))

    # 4. Завершение хода
    actions.append(('end_turn',))
    
    # Создаем словари для сопоставления
    action_to_index = {action: i for i, action in enumerate(actions)}
    index_to_action = {i: action for i, action in enumerate(actions)}
    
    return action_to_index, index_to_action

# Глобально создаем пространство действий при импорте модуля
ACTION_TO_INDEX, INDEX_TO_ACTION = get_action_space()
ACTION_SPACE_SIZE = len(ACTION_TO_INDEX)


# Словари для сопоставления направлений с индексами плоскостей в модели
MOVE_DIRECTIONS = {
    (-1, 0): 0,  # N
    (-1, 1): 1,  # NE
    (0, 1): 2,   # E
    (1, 1): 3,   # SE
    (1, 0): 4,   # S
    (1, -1): 5,  # SW
    (0, -1): 6,  # W
    (-1, -1): 7, # NW
}
# 8 плоскостей

MERGE_DIRECTIONS = {
    (0, 1): 8,   # E
    (1, 0): 9,   # S
    (1, 1): 10,  # SE
    (1, -1): 11, # SW (относительно первой клетки в отсортированной паре)
}
# 4 плоскости

# Индексы плоскостей для построек
BUILD_PEASANT_PLANE = 12
BUILD_WARRIOR_PLANE = 13

NUM_ACTION_PLANES = 14
