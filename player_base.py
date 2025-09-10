# player_base.py

class Player:
    """ Базовый класс для всех игроков. Не импортирует ничего лишнего. """
    def __init__(self, player_id):
        self.player_id = player_id
    
    def get_move(self, game_state, events, *args, **kwargs):
        raise NotImplementedError
