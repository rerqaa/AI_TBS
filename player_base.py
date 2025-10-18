class Player:
    def __init__(self, player_id):
        self.player_id = player_id
    
    def get_move(self, game_state, events, *args, **kwargs):
        raise NotImplementedError
