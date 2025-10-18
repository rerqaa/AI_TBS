import random
import torch
import torch.nn.functional as F
import numpy as np
from constants import *
from utils import INDEX_TO_ACTION, ACTION_TO_INDEX
from mcts import MCTS
from player_base import Player

class RandomBotPlayer(Player):
    def get_move(self, game_state, events, *args, **kwargs):
        moves = game_state.get_legal_moves()
        if len(moves) == 1 and moves[0] == ('end_turn',):
            return moves[0]
        else:
            non_end_moves = [m for m in moves if m != ('end_turn',)]
            if non_end_moves and random.random() > 0.1:
                return random.choice(non_end_moves)
            else:
                return ('end_turn',)

class AIPlayer(Player):
    def __init__(self, player_id, model, num_simulations=50, device="cpu", c_puct=1.0, mcts_batch_size=32):
        super().__init__(player_id)
        self.model = model
        self.num_simulations = num_simulations
        self.device = device
        self.c_puct = c_puct
        self.mcts_searcher = MCTS(self.model, c_puct=self.c_puct, device=self.device, mcts_batch_size=mcts_batch_size)

    def get_move(self, game_state, events, temperature=0.0, *args, **kwargs):
        self.model.eval()
        root_node = self.mcts_searcher.search(game_state, self.num_simulations)
        chosen_move, _ = self.mcts_searcher.get_action_probs(root_node, temperature=temperature)
        return chosen_move
