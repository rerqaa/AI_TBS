import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp
from multiprocessing import Pool
from torch.amp import GradScaler, autocast
import traceback
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from game_state import GameState
from model import StrategyNet
from mcts import MCTS
from bots import AIPlayer
from constants import *
from utils import ACTION_SPACE_SIZE, ACTION_TO_INDEX, SYMMETRY_LOOKUPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ITERATIONS = 1000
GAMES_PER_ITERATION = 40
EPOCHS_PER_ITERATION = 3
BATCH_SIZE = 256
LEARNING_RATES_TO_TRY = [5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5] 
WEIGHT_DECAY = 1e-4
REPLAY_BUFFER_SIZE = 500000
MIN_REPLAY_BUFFER_SIZE = 2000
NUM_WORKERS = 4 
GRADIENT_CLIP_NORM = 1.0
EMA_ALPHA = 0.2

MCTS_SIMULATIONS = 80
MCTS_BATCH_SIZE = 64
CPUCT = 1.5
TEMPERATURE = 1.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

EVAL_GAMES = 20
EVAL_WIN_THRESHOLD = 0.55
EVAL_MCTS_SIMS = 200
EVAL_TEMPERATURE = 0.1

MODEL_PATH = "best_model.pth"
CHECKPOINT_PATH = "latest_checkpoint.pth"

def self_play_worker(model_state_dict, device_str):
    try:
        device = torch.device(device_str)
        model = StrategyNet().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        game_history = []
        game_state = GameState()
        mcts = MCTS(model, CPUCT, device, mcts_batch_size=MCTS_BATCH_SIZE)
        mcts_root = None

        while game_state.get_game_ended() == 0:
            current_player = game_state.current_player
            
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                root = mcts.search(game_state, MCTS_SIMULATIONS, DIRICHLET_ALPHA, DIRICHLET_EPSILON, root=mcts_root)
            
            chosen_move, policy_target = mcts.get_action_probs(root, temperature=TEMPERATURE)
            
            observation = model.get_observation(game_state)
            spatial_cpu = observation[0].cpu()
            global_cpu = observation[1].cpu()
            game_history.append([(spatial_cpu, global_cpu), policy_target, current_player])
            
            game_state = game_state.apply_move(chosen_move)
            
            chosen_move_idx = ACTION_TO_INDEX.get(chosen_move)
            if chosen_move_idx in root.children:
                mcts_root = root.children[chosen_move_idx]
            else:
                mcts_root = None
            
        winner = game_state.get_game_ended()
        
        training_data = []
        for observation, policy_target, player_id in game_history:
            value = 0.0
            if winner != 1e-4:
                if winner == player_id: value = 1.0
                elif winner == -player_id: value = -1.0
            training_data.append((observation, policy_target, value))
        
        game_length = game_state.turn_count
        return training_data, game_length, winner

    except Exception as e:
        print(f"Ошибка в процессе self-play worker: {e}")
        traceback.print_exc()
        return [], 0, 0

def train_step(model, optimizer, replay_buffer, device, scaler):
    if len(replay_buffer) < BATCH_SIZE: return 0.0, 0.0, 0.0
    model.train()
    batch = random.sample(replay_buffer, BATCH_SIZE)
    observations, policies, values = zip(*batch)

    spatial_batch, global_batch = zip(*observations)
    spatial_tensors = torch.cat(spatial_batch)
    
    sym_indices = torch.randint(0, 8, (BATCH_SIZE,))
    
    policies_np = np.array(policies)
    augmented_policies_np = np.empty_like(policies_np)
    for i in range(BATCH_SIZE):
        lookup = SYMMETRY_LOOKUPS[sym_indices[i]]
        augmented_policies_np[i] = policies_np[i][lookup]
    policies_tensor = torch.from_numpy(augmented_policies_np).float().to(device, non_blocking=True)
    
    flip_mask = (sym_indices >= 4)
    spatial_tensors[flip_mask] = torch.flip(spatial_tensors[flip_mask], [3])
    
    rot_k_values = sym_indices % 4
    for k in range(1, 4):
        rot_mask = (rot_k_values == k)
        if rot_mask.any():
            spatial_tensors[rot_mask] = torch.rot90(spatial_tensors[rot_mask], k=k, dims=[2, 3])

    spatial_tensors = spatial_tensors.to(device, non_blocking=True)
    global_tensors = torch.cat(global_batch).to(device, non_blocking=True)
    values_tensor = torch.tensor(values, dtype=torch.float32).view(-1, 1).to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        policy_logits, value_preds = model(spatial_tensors, global_tensors)
        value_loss = F.mse_loss(value_preds, values_tensor)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_tensor * log_probs).sum(dim=1).mean()
        total_loss = value_loss + policy_loss
        
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    
    scaler.step(optimizer)
    scaler.update()
    
    return total_loss.item(), policy_loss.item(), value_loss.item()

def evaluate_models(model1_dict, model2_dict, device_str, num_games):
    device = torch.device(device_str)
    model1 = StrategyNet().to(device)
    model1.load_state_dict(model1_dict)
    model2 = StrategyNet().to(device)
    model2.load_state_dict(model2_dict)
    
    wins_model1 = 0; losses_model1 = 0; draws = 0
    
    for i in range(num_games):
        game_state = GameState()
        p1 = AIPlayer(PLAYER_1, model1, num_simulations=EVAL_MCTS_SIMS, device=device, c_puct=CPUCT, mcts_batch_size=MCTS_BATCH_SIZE)
        p2 = AIPlayer(PLAYER_2, model2, num_simulations=EVAL_MCTS_SIMS, device=device, c_puct=CPUCT, mcts_batch_size=MCTS_BATCH_SIZE)
        
        players = {PLAYER_1: p1, PLAYER_2: p2} if i % 2 == 0 else {PLAYER_1: p2, PLAYER_2: p1}
        model1_player_id = PLAYER_1 if i % 2 == 0 else PLAYER_2
        
        while game_state.get_game_ended() == 0:
            active_player_obj = players[game_state.current_player]
            move = active_player_obj.get_move(game_state, None, temperature=EVAL_TEMPERATURE)
            game_state = game_state.apply_move(move)

        winner = game_state.get_game_ended()
        
        if winner == 1e-4:
            draws += 1
        elif winner == model1_player_id:
            wins_model1 += 1
        else:
            losses_model1 += 1
            
    return wins_model1, losses_model1, draws

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    torch.backends.cudnn.benchmark = True
        
    print(f"Используемое устройство: {DEVICE}")
    print(f"Количество рабочих процессов: {NUM_WORKERS}")

    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = f"runs/alphazero_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard логи сохраняются в: {log_dir}")
    
    current_model = StrategyNet().to(DEVICE)
    best_model = StrategyNet().to(DEVICE)
    
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATES_TO_TRY[0], weight_decay=WEIGHT_DECAY)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    start_iteration = 0
    global_training_step = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Загрузка чекпоинта из {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        current_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        replay_buffer = deque(checkpoint['replay_buffer'], maxlen=REPLAY_BUFFER_SIZE)
        start_iteration = checkpoint['iteration'] + 1
        global_training_step = checkpoint.get('global_training_step', 0)
        
        if os.path.exists(MODEL_PATH):
            best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        else: # Fallback if best_model wasn't saved separately
            best_model.load_state_dict(checkpoint['best_model_state_dict'])
        print(f"Обучение возобновлено с итерации {start_iteration + 1}")
    elif os.path.exists(MODEL_PATH):
        print(f"Загрузка существующей лучшей модели из {MODEL_PATH}")
        best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        current_model.load_state_dict(best_model.state_dict())
    
    if writer:
        dummy_spatial = torch.randn(1, 6, BOARD_SIZE, BOARD_SIZE).to(DEVICE)
        dummy_global = torch.randn(1, 2).to(DEVICE)
        try:
            writer.add_graph(StrategyNet().to(DEVICE), (dummy_spatial, dummy_global))
        except Exception as e:
            print(f"Не удалось записать граф модели в TensorBoard: {e}")

    for iteration in range(start_iteration, NUM_ITERATIONS):
        print(f"\n--- Итерация {iteration + 1} / {NUM_ITERATIONS} ---")
        
        current_model.eval()
        
        current_cpu_weights = {k: v.cpu() for k, v in current_model.state_dict().items()}

        all_games_data = []
        game_lengths = []
        winner_stats = {PLAYER_1: 0, PLAYER_2: 0, 1e-4: 0}

        print(f"Фаза 1: Self-Play ({GAMES_PER_ITERATION} игр на {NUM_WORKERS} процессах)...")
        with Pool(processes=int(NUM_WORKERS)) as pool:
            async_results = [pool.apply_async(self_play_worker, args=(current_cpu_weights, str(DEVICE))) for _ in range(GAMES_PER_ITERATION)]
            for res in tqdm(async_results, total=GAMES_PER_ITERATION):
                game_data, game_length, winner = res.get()
                all_games_data.extend(game_data)
                game_lengths.append(game_length)
                if winner in winner_stats:
                    winner_stats[winner] += 1
                
        replay_buffer.extend(all_games_data)
        print(f"Размер буфера: {len(replay_buffer)} / {REPLAY_BUFFER_SIZE}")

        if writer and game_lengths:
            writer.add_scalar('SelfPlay/Avg_Game_Length', np.mean(game_lengths), iteration)
            writer.add_scalar('SelfPlay/Wins_P1', winner_stats[PLAYER_1] / GAMES_PER_ITERATION, iteration)
            writer.add_scalar('SelfPlay/Wins_P2', winner_stats[PLAYER_2] / GAMES_PER_ITERATION, iteration)
            writer.add_scalar('SelfPlay/Draws', winner_stats[1e-4] / GAMES_PER_ITERATION, iteration)
            writer.add_scalar('ReplayBuffer/Size', len(replay_buffer), iteration)
        
        if len(replay_buffer) < MIN_REPLAY_BUFFER_SIZE:
            print(f"Сбор данных... Пропуск фазы обучения (нужно {MIN_REPLAY_BUFFER_SIZE} примеров).")
            continue

        print("Фаза 2: Адаптивное обучение с поиском лучшего Learning Rate...")
        
        initial_weights = {k: v.clone() for k, v in current_model.state_dict().items()}
        
        best_model_cpu_weights = {k: v.cpu() for k, v in best_model.state_dict().items()}

        best_failure = {'win_rate': -1, 'weights': None, 'lr': None, 'loss': float('inf')}
        found_new_champion = False
        final_win_rate_this_iter = -1
        chosen_lr_this_iter = 0

        for lr in LEARNING_RATES_TO_TRY:
            print(f"\n--- Попытка с LR = {lr:.1e} ---")
            
            current_model.load_state_dict(initial_weights)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            total_loss_sum, num_batches = 0, 0
            steps = (len(replay_buffer) // BATCH_SIZE) * EPOCHS_PER_ITERATION
            
            pbar = tqdm(range(steps), desc=f"Обучение (LR={lr:.1e})")
            for _ in pbar:
                if len(replay_buffer) >= BATCH_SIZE:
                   t_loss, p_loss, v_loss = train_step(current_model, optimizer, replay_buffer, DEVICE, scaler)
                   if writer:
                       writer.add_scalar('Loss/Total_Step', t_loss, global_training_step)
                       writer.add_scalar('Loss/Policy_Step', p_loss, global_training_step)
                       writer.add_scalar('Loss/Value_Step', v_loss, global_training_step)
                       writer.add_scalar('Params/Learning_Rate_Step', lr, global_training_step)
                   global_training_step += 1
                   
                   total_loss_sum += t_loss
                   num_batches += 1
                   if num_batches > 0:
                       pbar.set_postfix({"avg_loss": f"{total_loss_sum / num_batches:.4f}"})
            
            avg_loss = total_loss_sum / num_batches if num_batches > 0 else float('inf')

            print("  Оценка обученной модели...")
            
            current_cpu_weights_for_eval = {k: v.cpu() for k, v in current_model.state_dict().items()}
            
            num_eval_workers = min(NUM_WORKERS, EVAL_GAMES)
            games_per_worker = [EVAL_GAMES // num_eval_workers] * num_eval_workers
            for i in range(EVAL_GAMES % num_eval_workers): games_per_worker[i] += 1

            with Pool(processes=num_eval_workers) as pool:
                async_results = [
                    pool.apply_async(evaluate_models, args=(current_cpu_weights_for_eval, best_model_cpu_weights, str(DEVICE), games))
                    for games in games_per_worker if games > 0
                ]
                total_wins, total_losses, total_draws = 0, 0, 0
                for res in async_results:
                    wins, losses, draws = res.get()
                    total_wins += wins
                    total_losses += losses
                    total_draws += draws
            
            total_eval_games = total_wins + total_losses
            win_rate = total_wins / total_eval_games if total_eval_games > 0 else 0
            print(f"  Результат против лучшей модели: {win_rate * 100:.1f}% побед ({total_wins}W / {total_losses}L / {total_draws}D).")

            if writer:
                hparams = {'lr': lr, 'iteration': iteration}
                metrics = {'hparam/eval_win_rate': win_rate, 'hparam/avg_train_loss': avg_loss}
                writer.add_hparams(hparams, metrics)

            if win_rate > EVAL_WIN_THRESHOLD:
                print(f"*** УСПЕХ! Найден новый чемпион с LR={lr}. ***")
                torch.save(current_model.state_dict(), MODEL_PATH)
                best_model.load_state_dict(current_model.state_dict())
                found_new_champion = True
                final_win_rate_this_iter = win_rate
                chosen_lr_this_iter = lr
                break
            else:
                if win_rate > best_failure['win_rate'] or (win_rate == best_failure['win_rate'] and avg_loss < best_failure['loss']):
                    best_failure['win_rate'] = win_rate; best_failure['weights'] = current_cpu_weights_for_eval
                    best_failure['lr'] = lr; best_failure['loss'] = avg_loss

        if not found_new_champion:
            print("\nНи одна из попыток не создала нового чемпиона.")
            if best_failure['weights'] is not None:
                print(f"Применяем EMA к лучшей из неудачных моделей (LR={best_failure['lr']:.1e}, WR={best_failure['win_rate']:.2f}).")
                final_win_rate_this_iter = best_failure['win_rate']; chosen_lr_this_iter = best_failure['lr']
                best_failure_weights_on_device = {k: v.to(DEVICE) for k, v in best_failure['weights'].items()}
                
                temp_model = StrategyNet().to(DEVICE)
                temp_model.load_state_dict(best_failure_weights_on_device)
                
                current_weights = temp_model.state_dict()
                best_weights = best_model.state_dict()
                ema_weights = {key: EMA_ALPHA * current_weights[key] + (1 - EMA_ALPHA) * best_weights[key] for key in best_weights}
                current_model.load_state_dict(ema_weights)
                print("EMA применена. Модель частично обновлена для следующего self-play.")
            else:
                final_win_rate_this_iter = 0; chosen_lr_this_iter = 0
                current_model.load_state_dict(initial_weights)

        if writer:
            writer.add_scalar('Evaluation/WinRate_vs_Best', final_win_rate_this_iter, iteration)
            writer.add_scalar('Params/Chosen_LR_per_Iteration', chosen_lr_this_iter, iteration)
            writer.flush()
        
        torch.save({
            'iteration': iteration, 'global_training_step': global_training_step,
            'model_state_dict': current_model.state_dict(),
            'best_model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'replay_buffer': list(replay_buffer)
        }, CHECKPOINT_PATH)
        print(f"Чекпоинт сохранен в {CHECKPOINT_PATH}")

    if writer:
        writer.close()
