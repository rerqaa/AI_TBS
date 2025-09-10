# mcts.py

import numpy as np
import math
import torch
import torch.nn.functional as F
from constants import *
from utils import ACTION_SPACE_SIZE, ACTION_TO_INDEX, INDEX_TO_ACTION
from torch.amp import autocast

class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.q_value = 0.0
        self.prior_prob = prior_p

    def expand(self, policy, legal_moves_indices):
        for action_idx in legal_moves_indices:
            if action_idx not in self.children:
                self.children[action_idx] = Node(self, policy[action_idx])

    def select(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        sqrt_total_visits = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            q = -child.q_value / child.visit_count if child.visit_count > 0 else 0.0
            u = c_puct * child.prior_prob * sqrt_total_visits / (1 + child.visit_count)
            score = q + u
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def is_leaf(self):
        return len(self.children) == 0
        
    def backpropagate(self, value):
        self.visit_count += 1
        self.q_value += value
        if self.parent:
            self.parent.backpropagate(-value)


class MCTS:
    # <<< ИЗМЕНЕНИЕ: Добавляем mcts_batch_size в конструктор >>>
    def __init__(self, model, c_puct=1.0, device="cpu", mcts_batch_size=32):
        self.model = model
        self.c_puct = c_puct
        self.device = device
        self.mcts_batch_size = mcts_batch_size

    def _traverse_to_leaf(self, start_node, game_state):
        node = start_node
        temp_state = game_state.clone()
        
        while not node.is_leaf():
            action_idx, node = node.select(self.c_puct)
            if action_idx == -1: break
            move = INDEX_TO_ACTION[action_idx]
            temp_state = temp_state.apply_move(move)
        
        return node, temp_state

    def search(self, game_state, num_simulations, dirichlet_alpha=None, dirichlet_epsilon=0.25):
        root = Node(None, 1.0)
        
        # <<< ИЗМЕНЕНИЕ: Добавляем non_blocking=True для асинхронной передачи >>>
        board_tensor = self.model.get_observation(game_state).to(self.device, non_blocking=True)
        with torch.no_grad(), autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
            policy_logits, value = self.model(board_tensor)
        
        root.backpropagate(value.item())

        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        legal_moves = game_state.get_legal_moves()
        legal_moves_indices = [ACTION_TO_INDEX[m] for m in legal_moves if m in ACTION_TO_INDEX]
        
        mask = np.zeros_like(policy_probs)
        if legal_moves_indices:
            mask[legal_moves_indices] = 1
        policy_probs *= mask
        
        sum_probs = np.sum(policy_probs)
        if sum_probs > 1e-6:
            policy_probs /= sum_probs
        elif legal_moves_indices:
            prob = 1.0 / len(legal_moves_indices)
            for i in legal_moves_indices: policy_probs[i] = prob
        
        if dirichlet_alpha is not None and legal_moves_indices:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves_indices))
            valid_indices = np.array(legal_moves_indices)
            policy_probs[valid_indices] = (1 - dirichlet_epsilon) * policy_probs[valid_indices] + dirichlet_epsilon * noise

        root.expand(policy_probs, legal_moves_indices)

        completed_simulations = 0
        while completed_simulations < num_simulations:
            batch_requests = []
            
            # <<< ИЗМЕНЕНИЕ: Используем self.mcts_batch_size >>>
            for _ in range(self.mcts_batch_size):
                if completed_simulations >= num_simulations: break

                leaf_node, temp_state = self._traverse_to_leaf(root, game_state)
                
                winner = temp_state.get_game_ended()
                if winner != 0:
                    value_for_player_who_moved = winner * -temp_state.current_player
                    if winner == 1e-4:
                        value_for_player_who_moved = 0.0
                    
                    leaf_node.backpropagate(value_for_player_who_moved)
                    completed_simulations += 1
                else:
                    batch_requests.append({'node': leaf_node, 'state': temp_state})

            if not batch_requests:
                continue

            # <<< ИЗМЕНЕНИЕ: Добавляем non_blocking=True для асинхронной передачи >>>
            board_tensors = torch.cat(
                [self.model.get_observation(req['state']) for req in batch_requests]
            ).to(self.device, non_blocking=True)

            with torch.no_grad(), autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                policy_logits_batch, value_batch = self.model(board_tensors)
            
            policy_probs_batch = F.softmax(policy_logits_batch, dim=1).cpu().numpy()
            values = value_batch.cpu().numpy().flatten()

            for i, req in enumerate(batch_requests):
                node_to_expand = req['node']
                state_to_expand = req['state']
                policy_probs = policy_probs_batch[i]
                value = values[i]

                legal_moves = state_to_expand.get_legal_moves()
                legal_moves_indices = [ACTION_TO_INDEX.get(m) for m in legal_moves if m in ACTION_TO_INDEX]

                if legal_moves_indices:
                    mask = np.zeros_like(policy_probs)
                    mask[legal_moves_indices] = 1
                    policy_probs *= mask
                    sum_probs = np.sum(policy_probs)
                    if sum_probs > 1e-6:
                        policy_probs /= sum_probs
                    else:
                        prob = 1.0 / len(legal_moves_indices)
                        for idx in legal_moves_indices: policy_probs[idx] = prob
                    
                    node_to_expand.expand(policy_probs, legal_moves_indices)

                node_to_expand.backpropagate(value)
                completed_simulations += 1
                
        return root

    def get_action_probs(self, root, temperature=1.0):
        if not root.children: 
            legal_moves = [('end_turn',)]
            end_turn_idx = ACTION_TO_INDEX.get(legal_moves[0])
            pi = np.zeros(ACTION_SPACE_SIZE)
            if end_turn_idx is not None:
                pi[end_turn_idx] = 1.0
            return legal_moves[0], pi

        children_items = sorted(root.children.items())
        visit_counts = np.array([child.visit_count for _, child in children_items])
        action_indices = np.array([action for action, _ in children_items])
        
        if np.sum(visit_counts) == 0:
             chosen_action = np.random.choice(action_indices)
             pi = np.zeros(ACTION_SPACE_SIZE)
             pi[action_indices] = 1.0 / len(action_indices)
             return INDEX_TO_ACTION[chosen_action], pi

        if temperature == 0:
            best_action_idx_in_array = np.argmax(visit_counts)
            best_action = action_indices[best_action_idx_in_array]
            pi = np.zeros(ACTION_SPACE_SIZE)
            pi[best_action] = 1.0
            return INDEX_TO_ACTION[best_action], pi

        visit_counts_temp = visit_counts**(1.0 / temperature)
        probs = visit_counts_temp / np.sum(visit_counts_temp)
        chosen_action_idx_in_array = np.random.choice(len(probs), p=probs)
        chosen_action = action_indices[chosen_action_idx_in_array]
        
        pi = np.zeros(ACTION_SPACE_SIZE)
        for i, action_idx in enumerate(action_indices):
            pi[action_idx] = probs[i]
            
        return INDEX_TO_ACTION[chosen_action], pi
