# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from utils import (
    ACTION_SPACE_SIZE, ACTION_TO_INDEX, INDEX_TO_ACTION,
    MOVE_DIRECTIONS, MERGE_DIRECTIONS,
    BUILD_PEASANT_PLANE, BUILD_WARRIOR_PLANE, 
    # END_TURN_PLANE, # УДАЛЕНО
    NUM_ACTION_PLANES
)

class ConvolutionalBlock(nn.Module):
    """
    Полноценный остаточный блок в стиле ResNet.
    Структура: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
    """
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
        # Прибавляем residual ПЕРЕД финальной активацией
        out += residual
        return F.relu(out)

class StrategyNet(nn.Module):
    """
    Основная архитектура нейросети в стиле AlphaZero с остаточными блоками
    и пространственной головой политики.
    """
    def __init__(self, num_input_channels=6, num_blocks=5, num_filters=64):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        
        self.residual_blocks = nn.ModuleList([ConvolutionalBlock(num_filters) for _ in range(num_blocks)])
        
        # --- ГОЛОВА ПОЛИТИКИ ---
        # 1. Сверточная часть для пространственных действий
        self.policy_conv = nn.Conv2d(num_filters, NUM_ACTION_PLANES, kernel_size=1)
        # 2. Полносвязная часть для НЕпространственных действий (end_turn)
        self.policy_fc_end_turn = nn.Linear(num_filters, 1)

        # --- ГОЛОВА ЦЕННОСТИ ---
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 64)
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
                # Для полносвязных слоев лучше подходит стандартная инициализация Pytorch
                # или нормальное распределение с небольшим std.
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # Метод _create_action_map остается почти без изменений
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
        

    def forward(self, x):
        # Общая "башня"
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        # --- Голова Политики ---
        batch_size = x.size(0)
        policy_logits = torch.zeros(batch_size, ACTION_SPACE_SIZE, device=x.device)

        # 1. Обработка пространственных действий
        policy_planes = self.policy_conv(x)
        
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        values_to_put = policy_planes[batch_indices, self.map_plane_indices, self.map_r_indices, self.map_c_indices]
        policy_logits[batch_indices, self.map_action_indices] = values_to_put.float()
        
        # 2. ### НОВАЯ, ПРАВИЛЬНАЯ ЛОГИКА ДЛЯ end_turn ###
        # Усредняем пространственные признаки до одного вектора
        avg_features = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
        # Пропускаем через полносвязный слой
        end_turn_logit = self.policy_fc_end_turn(avg_features)
        # Вставляем полученный логит в общий вектор
        policy_logits[:, self.end_turn_action_idx] = end_turn_logit.squeeze(-1).float()
        
        # --- Голова Ценности ---
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    # Метод get_observation остается без изменений
    def get_observation(self, game_state):
        my_peasants = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        my_warriors = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        my_territory = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        enemy_peasants = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        enemy_warriors = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        enemy_territory = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        p = game_state.current_player
        if p == PLAYER_1:
            my_peasants[game_state.units_board == PEASANT_P1] = 1
            my_warriors[game_state.units_board == WARRIOR_P1] = 1
            my_territory[game_state.territory_board == PLAYER_1] = 1
            enemy_peasants[game_state.units_board == PEASANT_P2] = 1
            enemy_warriors[game_state.units_board == WARRIOR_P2] = 1
            enemy_territory[game_state.territory_board == PLAYER_2] = 1
        else:
            my_peasants[game_state.units_board == PEASANT_P2] = 1
            my_warriors[game_state.units_board == WARRIOR_P2] = 1
            my_territory[game_state.territory_board == PLAYER_2] = 1
            enemy_peasants[game_state.units_board == PEASANT_P1] = 1
            enemy_warriors[game_state.units_board == WARRIOR_P1] = 1
            enemy_territory[game_state.territory_board == PLAYER_1] = 1
            
        state_tensor = np.stack([
            my_peasants, my_warriors, my_territory,
            enemy_peasants, enemy_warriors, enemy_territory
        ])
        
        return torch.from_numpy(state_tensor).float().unsqueeze(0)
