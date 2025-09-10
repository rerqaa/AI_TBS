# train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp
from multiprocessing import Pool
from torch.amp import GradScaler, autocast

from game_state import GameState
from model import StrategyNet
from mcts import MCTS
from bots import AIPlayer
from constants import *
from utils import ACTION_SPACE_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Гиперпараметры ---
NUM_ITERATIONS = 1000
GAMES_PER_ITERATION = 50
EPOCHS_PER_ITERATION = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
REPLAY_BUFFER_SIZE = 500000
MIN_REPLAY_BUFFER_SIZE = 2000
# <<< ИЗМЕНЕНИЕ: Увеличиваем количество воркеров для лучшей утилизации CPU >>>
NUM_WORKERS = 6 
GRADIENT_CLIP_NORM = 1.0

# --- Гиперпараметры MCTS ---
MCTS_SIMULATIONS = 100
# <<< ИЗМЕНЕНИЕ: Увеличиваем размер батча для лучшей утилизации GPU >>>
MCTS_BATCH_SIZE = 64
CPUCT = 1.5
TEMPERATURE = 1.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# --- Параметры оценки модели ---
EVAL_GAMES = 2
EVAL_WIN_THRESHOLD = 0.55
EVAL_MCTS_SIMS = 100
EVAL_TEMPERATURE = 0.0

MODEL_PATH = "best_model.pth"
EVAL_LOG_DIR = "eval_games"

def get_board_string(game_state):
    UNIT_TO_CHAR = {
        EMPTY: '.', PEASANT_P1: 'p', WARRIOR_P1: 'W',
        PEASANT_P2: 'k', WARRIOR_P2: 'R'
    }
    board_str = "  " + "".join([f"{i:<2}" for i in range(BOARD_SIZE)]) + "\n"
    for r in range(BOARD_SIZE):
        row_str = f"{r:<2}"
        for c in range(BOARD_SIZE):
            unit = game_state.units_board[r, c]
            char = UNIT_TO_CHAR.get(unit, '?')
            row_str += f"{char} "
        board_str += row_str + "\n"
    return board_str

def self_play_worker(model_state_dict, device_str):
    device = torch.device(device_str)
    model = StrategyNet().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    game_history = []
    game_state = GameState()
    # <<< ИЗМЕНЕНИЕ: Передаем новый параметр в MCTS >>>
    mcts = MCTS(model, CPUCT, device, mcts_batch_size=MCTS_BATCH_SIZE)
    while game_state.get_game_ended() == 0:
        current_player = game_state.current_player
        board_tensor = model.get_observation(game_state).to(device)
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            root = mcts.search(game_state, MCTS_SIMULATIONS, DIRICHLET_ALPHA, DIRICHLET_EPSILON)
        chosen_move, policy_target = mcts.get_action_probs(root, temperature=TEMPERATURE)
        game_history.append([board_tensor.cpu(), policy_target, current_player])
        game_state = game_state.apply_move(chosen_move)
    winner = game_state.get_game_ended()
    training_data = []
    for board_tensor, policy_target, player_id in game_history:
        value = 0.0
        if winner != 1e-4:
            if winner == player_id: value = 1.0
            elif winner == -player_id: value = -1.0
        training_data.append((board_tensor, policy_target, value))
    return training_data

def train_step(model, optimizer, scheduler, replay_buffer, device, scaler):
    if len(replay_buffer) < BATCH_SIZE: return 0.0, 0.0, 0.0
    model.train()
    batch = random.sample(replay_buffer, BATCH_SIZE)
    boards, policies, values = zip(*batch)
    boards_tensor = torch.cat(boards, dim=0).to(device, non_blocking=True)
    policies_tensor = torch.tensor(np.array(policies), dtype=torch.float32).to(device, non_blocking=True)
    values_tensor = torch.tensor(values, dtype=torch.float32).view(-1, 1).to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        policy_logits, value_preds = model(boards_tensor)
        value_loss = F.mse_loss(value_preds, values_tensor)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_tensor * log_probs).sum(dim=1).mean()
        total_loss = value_loss + policy_loss
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    
    # <<< ИЗМЕНЕНИЕ: Правильный порядок вызовов >>>
    scaler.step(optimizer)
    scaler.update()
    scheduler.step() # Теперь scheduler.step() вызывается ПОСЛЕ optimizer.step()
    
    return total_loss.item(), policy_loss.item(), value_loss.item()

def evaluate_models(model1_dict, model2_dict, device_str, iteration_num):
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    device = torch.device(device_str)
    model1 = StrategyNet().to(device)
    model1.load_state_dict(model1_dict)
    model2 = StrategyNet().to(device)
    model2.load_state_dict(model2_dict)
    wins_model1 = 0; losses_model1 = 0; draws = 0
    
    for i in range(EVAL_GAMES):
        game_state = GameState()
        # <<< ИЗМЕНЕНИЕ: Передаем mcts_batch_size и в оценку >>>
        p1 = AIPlayer(PLAYER_1, model1, num_simulations=EVAL_MCTS_SIMS, device=device, c_puct=CPUCT, mcts_batch_size=MCTS_BATCH_SIZE)
        p2 = AIPlayer(PLAYER_2, model2, num_simulations=EVAL_MCTS_SIMS, device=device, c_puct=CPUCT, mcts_batch_size=MCTS_BATCH_SIZE)
        
        players = {PLAYER_1: p1, PLAYER_2: p2} if i % 2 == 0 else {PLAYER_1: p2, PLAYER_2: p1}
        model1_player_id = PLAYER_1 if i % 2 == 0 else PLAYER_2
        
        game_log = [f"Iteration: {iteration_num}, Game: {i}", f"Model_1 (current) plays as Player {model1_player_id}\n"]

        while game_state.get_game_ended() == 0:
            active_player_obj = players[game_state.current_player]
            player_name = "Model_1 (current)" if active_player_obj == p1 else "Model_2 (best)"
            move = active_player_obj.get_move(game_state, None, temperature=EVAL_TEMPERATURE)
            log_entry = f"Turn {game_state.turn_count}, Player {game_state.current_player} ({player_name}) chooses: {move}"
            game_log.append(log_entry)
            game_state = game_state.apply_move(move)
            game_log.append(get_board_string(game_state))

        winner = game_state.get_game_ended()
        game_log.append(f"\n--- GAME OVER ---")
        
        if winner == 1e-4:
            draws += 1; game_log.append("Result: Draw")
        elif winner == model1_player_id:
            wins_model1 += 1; game_log.append("Result: Model_1 (current) WINS")
        else:
            losses_model1 += 1; game_log.append("Result: Model_2 (best) WINS")
        
        log_filename = f"iter_{iteration_num}_game_{i}.txt"
        with open(os.path.join(EVAL_LOG_DIR, log_filename), 'w', encoding='utf-8') as f:
            f.write("\n".join(game_log))
            
    return wins_model1, losses_model1, draws

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    print(f"Используемое устройство: {DEVICE}")
    print(f"Количество рабочих процессов: {NUM_WORKERS}")
    current_model = StrategyNet().to(DEVICE); best_model = StrategyNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Загрузка существующей лучшей модели из {MODEL_PATH}")
        best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    current_model.load_state_dict(best_model.state_dict())
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS * (REPLAY_BUFFER_SIZE // BATCH_SIZE) * EPOCHS_PER_ITERATION)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Итерация {iteration + 1} / {NUM_ITERATIONS} ---")
        current_model.eval()
        current_cpu_weights = {k: v.cpu() for k, v in current_model.state_dict().items()}
        all_games_data = []
        print(f"Фаза 1: Self-Play ({GAMES_PER_ITERATION} игр на {NUM_WORKERS} процессах)...")
        with Pool(processes=int(NUM_WORKERS)) as pool:
            async_results = [pool.apply_async(self_play_worker, args=(current_cpu_weights, str(DEVICE))) for _ in range(GAMES_PER_ITERATION)]
            for res in tqdm(async_results, total=GAMES_PER_ITERATION):
                all_games_data.extend(res.get())
        replay_buffer.extend(all_games_data)
        print(f"Размер буфера: {len(replay_buffer)} / {REPLAY_BUFFER_SIZE}")
        if len(replay_buffer) < MIN_REPLAY_BUFFER_SIZE:
            print(f"Сбор данных... Пропуск фазы обучения (нужно {MIN_REPLAY_BUFFER_SIZE} примеров).")
            continue
        print(f"Фаза 2: Обучение ({EPOCHS_PER_ITERATION} эпох)...")
        total_loss_sum, policy_loss_sum, value_loss_sum = 0, 0, 0
        num_batches = 0
        steps = (len(replay_buffer) // BATCH_SIZE) * EPOCHS_PER_ITERATION
        for _ in tqdm(range(steps)):
            if len(replay_buffer) >= BATCH_SIZE:
               t_loss, p_loss, v_loss = train_step(current_model, optimizer, scheduler, replay_buffer, DEVICE, scaler)
               total_loss_sum += t_loss; policy_loss_sum += p_loss; value_loss_sum += v_loss
               num_batches += 1
        if num_batches > 0:
            print(f"Средняя общая потеря: {total_loss_sum / num_batches:.4f} | "
                  f"Политика: {policy_loss_sum / num_batches:.4f} | "
                  f"Ценность: {value_loss_sum / num_batches:.4f}")
            print(f"Текущая скорость обучения: {scheduler.get_last_lr()[0]:.8f}")
        print("Фаза 3: Оценка...")
        current_cpu_weights = {k: v.cpu() for k, v in current_model.state_dict().items()}
        best_cpu_weights = {k: v.cpu() for k, v in best_model.state_dict().items()}
        with Pool(processes=1) as pool:
            wins, losses, draws = pool.apply(evaluate_models, args=(current_cpu_weights, best_cpu_weights, str(DEVICE), iteration + 1))
        total_eval_games = wins + losses
        win_rate = wins / total_eval_games if total_eval_games > 0 else 0
        print(f"Результат против лучшей модели: {win_rate * 100:.1f}% побед ({wins}W / {losses}L / {draws}D).")
        if win_rate > EVAL_WIN_THRESHOLD:
            print(f"*** Новая лучшая модель найдена! Сохранение в {MODEL_PATH} ***")
            torch.save(current_model.state_dict(), MODEL_PATH)
            best_model.load_state_dict(current_model.state_dict())
        else:
            print("Новая модель не превзошла старую. Откат весов.")
            current_model.load_state_dict(best_model.state_dict())
