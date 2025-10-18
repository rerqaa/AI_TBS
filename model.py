import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from utils import (
    ACTION_SPACE_SIZE, ACTION_TO_INDEX, INDEX_TO_ACTION,
    MOVE_DIRECTIONS, MERGE_DIRECTIONS,
    BUILD_PEASANT_PLANE, BUILD_WARRIOR_PLANE, 
    END_TURN_PLANE,
    NUM_ACTION_PLANES
)

class ConvolutionalBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class StrategyNet(nn.Module):
    def __init__(self, num_input_channels=6, num_blocks=10, num_filters=128, num_global_features=2):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        
        self.residual_blocks = nn.ModuleList([ConvolutionalBlock(num_filters) for _ in range(num_blocks)])
        
        self.policy_conv = nn.Conv2d(num_filters, NUM_ACTION_PLANES, kernel_size=1)
        
        self.value_fc1 = nn.Linear(num_filters + num_global_features, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self._create_action_map()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _create_action_map(self):
        map_plane_indices = []
        map_r_indices = []
        map_c_indices = []
        map_action_indices = []

        for action, action_idx in ACTION_TO_INDEX.items():
            move_type = action[0]
            
            if move_type == 'move':
                from_pos, to_pos = action[1], action[2]
                dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
                plane_idx = MOVE_DIRECTIONS.get((dr, dc))
                if plane_idx is not None:
                    map_plane_indices.append(plane_idx); map_r_indices.append(from_pos[0])
                    map_c_indices.append(from_pos[1]); map_action_indices.append(action_idx)

            elif move_type == 'merge':
                pos1, pos2 = action[1][0], action[1][1]
                dr, dc = pos2[0] - pos1[0], pos2[1] - pos1[1]
                plane_idx = MERGE_DIRECTIONS.get((dr, dc))
                if plane_idx is not None:
                    map_plane_indices.append(plane_idx); map_r_indices.append(pos1[0])
                    map_c_indices.append(pos1[1]); map_action_indices.append(action_idx)

            elif move_type == 'build_peasant':
                pos = action[1]
                map_plane_indices.append(BUILD_PEASANT_PLANE); map_r_indices.append(pos[0])
                map_c_indices.append(pos[1]); map_action_indices.append(action_idx)

            elif move_type == 'build_warrior':
                pos = action[1]
                map_plane_indices.append(BUILD_WARRIOR_PLANE); map_r_indices.append(pos[0])
                map_c_indices.append(pos[1]); map_action_indices.append(action_idx)
        
        self.end_turn_action_idx = ACTION_TO_INDEX[('end_turn',)]
        self.register_buffer('map_plane_indices', torch.LongTensor(map_plane_indices))
        self.register_buffer('map_r_indices', torch.LongTensor(map_r_indices))
        self.register_buffer('map_c_indices', torch.LongTensor(map_c_indices))
        self.register_buffer('map_action_indices', torch.LongTensor(map_action_indices))
        

    def forward(self, spatial_input, global_input):
        x = F.relu(self.initial_bn(self.initial_conv(spatial_input)))
        for block in self.residual_blocks:
            x = block(x)

        batch_size = x.size(0)
        
        policy_logits = torch.zeros(batch_size, ACTION_SPACE_SIZE, device=x.device)
        policy_planes = self.policy_conv(x)
        
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        values_to_put = policy_planes[batch_indices, self.map_plane_indices, self.map_r_indices, self.map_c_indices]
        policy_logits[batch_indices, self.map_action_indices] = values_to_put.float()
        
        end_turn_logit = policy_planes[:, END_TURN_PLANE, 0, 0]
        policy_logits[:, self.end_turn_action_idx] = end_turn_logit.float()
        
        spatial_features = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        combined_features = torch.cat([spatial_features, global_input], dim=1)
        value = F.relu(self.value_fc1(combined_features))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def get_observation(self, game_state):
        spatial_tensor = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        p = game_state.current_player
        opponent = -p
        units_board = game_state.units_board
        territory_board = game_state.territory_board

        if p == PLAYER_1:
            spatial_tensor[0, units_board == PEASANT_P1] = 1
            spatial_tensor[1, units_board == WARRIOR_P1] = 1
            spatial_tensor[2, territory_board == PLAYER_1] = 1
            spatial_tensor[3, units_board == PEASANT_P2] = 1
            spatial_tensor[4, units_board == WARRIOR_P2] = 1
            spatial_tensor[5, territory_board == PLAYER_2] = 1
        else: # p == PLAYER_2
            spatial_tensor[0, units_board == PEASANT_P2] = 1
            spatial_tensor[1, units_board == WARRIOR_P2] = 1
            spatial_tensor[2, territory_board == PLAYER_2] = 1
            spatial_tensor[3, units_board == PEASANT_P1] = 1
            spatial_tensor[4, units_board == WARRIOR_P1] = 1
            spatial_tensor[5, territory_board == PLAYER_1] = 1
        
        MAX_COINS_FOR_NORMALIZATION = 1000.0
        my_coins = game_state.player_coins[p]
        opponent_coins = game_state.player_coins[opponent]
        
        global_features = np.array([
            min(my_coins, MAX_COINS_FOR_NORMALIZATION) / MAX_COINS_FOR_NORMALIZATION,
            min(opponent_coins, MAX_COINS_FOR_NORMALIZATION) / MAX_COINS_FOR_NORMALIZATION
        ], dtype=np.float32)
            
        return (
            torch.from_numpy(spatial_tensor).float().unsqueeze(0),
            torch.from_numpy(global_features).float().unsqueeze(0)
        )
