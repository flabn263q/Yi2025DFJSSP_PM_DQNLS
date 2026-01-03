import matplotlib
matplotlib.use('Agg') # 強制不顯示視窗，直接存檔

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque, namedtuple
import copy
import time
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

# ==========================================
# 1. Configuration & Parameters
# ==========================================
class Config:
    # --- Global Experiment Settings ---
    REPLICAS = 20           
    TRAIN_EPISODES = 2000   
    SEED = 53
    
    # --- Scales & Cases ---
    SCALES = [(6, 6), (15, 8), (20, 10)] # p13: Table3.
    CASE_SEEDS = [53, 54, 55, 56, 57] 
    
    # --- Job Parameters ---
    NUM_OPS_PER_JOB = 6 # p13: Table3.
    PROC_TIME_RANGE = (1, 20) # p13: Table3.
    
    # --- Default Constraints ---
    DEFAULT_Q = 3
    DEFAULT_R_III = 0.80
    R_II = 0.95 # p13: Table3.
    
    # --- Weibull Parameters ---
    WEIBULL_SHAPE_RANGE = (1.60, 1.80) # p13: Table3. 但論文中筆誤成scale範圍
    WEIBULL_SCALE_RANGE = (70.00, 78.00) # p13: Table3. 但論文中筆誤成shape範圍
    SIGMA = 0.3 # p + sigma * (age - t_II)，但論文中未明確指出sigma值，假設為0.3
    
    # --- Maintenance Specs ---
    DUR_U, DUR_V, DUR_W = 5, 10, 30 # p13: Table3.
    FACTOR_U, FACTOR_V, FACTOR_W = 0.65, 0.90, 0.50 # p9, p13:Table3.
    
    # --- RL Hyperparameters ---
    LR = 1e-4 # p13: Table3.是1e-5，但1e-4加速收斂
    GAMMA = 0.9 # p13: Table3.
    BUFFER_SIZE = 2000 # p13: Table3.是1000，但2000提高樣本多樣性
    BATCH_SIZE = 128 # p13: Table3.
    EPSILON_START = 1.0 # 論文未提及，採用Standard Practice
    EPSILON_END = 0.05 # 論文未提及，採用Standard Practice
    EPSILON_DECAY = 2000 # 論文未提及，採用Standard Practice
    TARGET_UPDATE = 50 # p13: Table3.
    HIDDEN_DIM = 128 # 論文未提及，採用標準MLP大小

# ==========================================
# 2. Core Classes (Job, Machine, STW)
# ==========================================
class Job:
    def __init__(self, job_id, ops_times):
        self.id = job_id
        self.ops_times = ops_times
        self.num_ops = len(ops_times)
        self.current_op_idx = 0
        self.next_ready_time = 0
        self.finished = False
        self.completion_time = 0

    def get_remaining_proc_time(self):
        if self.finished: return 0
        return sum(self.ops_times[self.current_op_idx:])

    def get_remaining_ops(self):
        return self.num_ops - self.current_op_idx

class Machine:
    def __init__(self, machine_id, beta, eta):
        self.id = machine_id
        self.beta = beta
        self.eta = eta
        self.age = 0.0
        self.available_time = 0.0
        self.history = []

    def get_reliability(self, added_age=0):
        t = self.age + added_age
        return math.exp(- (t / self.eta) ** self.beta)

    def calculate_actual_time(self, nominal_time, r_ii):
        r_curr = self.get_reliability()
        if r_curr > r_ii:
            return nominal_time
        else:
            t_II = self.eta * ((-math.log(r_ii)) ** (1/self.beta))
            penalty = Config.SIGMA * (self.age - t_II)
            return nominal_time + max(0, penalty)

class STW_Manager:
    def __init__(self, q_limit):
        self.q_limit = q_limit
        self.global_maintenance_log = []

    def find_slot(self, earliest_start, duration):
        candidate_start = earliest_start
        self.global_maintenance_log.sort(key=lambda x: x[0])
        while True:
            candidate_end = candidate_start + duration
            overlaps = [x for x in self.global_maintenance_log if max(candidate_start, x[0]) < min(candidate_end, x[1])]
            if len(overlaps) < self.q_limit:
                self.global_maintenance_log.append((candidate_start, candidate_end))
                return candidate_start, candidate_end
            else:
                overlaps.sort(key=lambda x: x[1])
                candidate_start = overlaps[0][1]

    def reset(self):
        self.global_maintenance_log = []

# ==========================================
# 3. RL Environment
# ==========================================
class DFJSP_Env(gym.Env):
    def __init__(self, num_jobs, num_machines, q_limit, r_iii):
        super(DFJSP_Env, self).__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.q_limit = q_limit
        self.r_iii = r_iii
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        self.stw = STW_Manager(self.q_limit)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.machines = []
        for i in range(self.num_machines):
            beta = np.random.uniform(*Config.WEIBULL_SHAPE_RANGE)
            eta = np.random.uniform(*Config.WEIBULL_SCALE_RANGE)
            self.machines.append(Machine(i, beta, eta))
        self.jobs = []
        for i in range(self.num_jobs):
            ops = [np.random.randint(*Config.PROC_TIME_RANGE) for _ in range(Config.NUM_OPS_PER_JOB)]
            self.jobs.append(Job(i, ops))
        self.stw.reset()
        self.finished_jobs = 0
        return self._get_state(), {}

    def _get_state(self):
        unfinished = [j for j in self.jobs if not j.finished]
        art_ave = np.mean([j.get_remaining_proc_time() for j in unfinished]) if unfinished else 0
        makespan = max([m.available_time for m in self.machines]) if self.machines else 1
        makespan = max(makespan, 1)
        utils = [sum([x[3]-x[2] for x in m.history if x[4]=='JOB']) / makespan for m in self.machines]
        urm_ave = np.mean(utils)
        urm_std = np.std(utils)
        comp_rates = [j.current_op_idx / j.num_ops for j in self.jobs]
        crw_ave = np.mean(comp_rates)
        crw_std = np.std(comp_rates)
        ma_ave = np.mean([m.age for m in self.machines])
        return np.array([art_ave, urm_ave, urm_std, crw_ave, crw_std, ma_ave], dtype=np.float32)

    def step(self, action):
        maint_decision = action // 5 
        rule_idx = action % 5        
        unfinished_jobs = [j for j in self.jobs if not j.finished]
        if not unfinished_jobs: return self._get_state(), 0, True, False, {}
        
        if rule_idx == 0: selected_job = min(unfinished_jobs, key=lambda j: j.get_remaining_proc_time())
        elif rule_idx == 1: selected_job = max(unfinished_jobs, key=lambda j: j.get_remaining_proc_time())
        elif rule_idx == 2: selected_job = max(unfinished_jobs, key=lambda j: j.get_remaining_ops())
        elif rule_idx == 3: selected_job = min(unfinished_jobs, key=lambda j: j.get_remaining_ops())
        elif rule_idx == 4: selected_job = random.choice(unfinished_jobs)
            
        selected_machine = min(self.machines, key=lambda m: m.available_time)
        prev_makespan = max([m.available_time for m in self.machines])
        
        r_curr = selected_machine.get_reliability()
        final_maint_type = 'Z'
        duration, factor = 0, 0
        
        if r_curr <= self.r_iii:
            final_maint_type, duration, factor = 'W', Config.DUR_W, Config.FACTOR_W
        elif maint_decision == 1: 
            final_maint_type, duration, factor = 'U', Config.DUR_U, Config.FACTOR_U
        elif maint_decision == 2: 
            final_maint_type, duration, factor = 'V', Config.DUR_V, Config.FACTOR_V
            
        if final_maint_type != 'Z':
            m_start = selected_machine.available_time
            real_start, real_end = self.stw.find_slot(m_start, duration)
            selected_machine.available_time = real_end
            selected_machine.age = selected_machine.age * (1.0 - factor)
            selected_machine.history.append((-1, -1, real_start, real_end, f'Maint_{final_maint_type}'))
            
        nominal_time = selected_job.ops_times[selected_job.current_op_idx]
        actual_time = selected_machine.calculate_actual_time(nominal_time, Config.R_II)
        start_time = max(selected_machine.available_time, selected_job.next_ready_time)
        end_time = start_time + actual_time
        selected_machine.available_time = end_time
        selected_machine.age += actual_time 
        selected_machine.history.append((selected_job.id, selected_job.current_op_idx, start_time, end_time, 'JOB'))
        selected_job.next_ready_time = end_time
        selected_job.current_op_idx += 1
        if selected_job.current_op_idx >= selected_job.num_ops:
            selected_job.finished = True
            selected_job.completion_time = end_time
            self.finished_jobs += 1
            
        new_makespan = max([m.available_time for m in self.machines])
        reward = -(new_makespan - prev_makespan)
        if final_maint_type == 'U': reward -= 10
        if final_maint_type == 'V': reward -= 20
        done = (self.finished_jobs == self.num_jobs)
        return self._get_state(), reward, done, False, {}

# ==========================================
# 4. DQN & Local Search
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, output_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    def push(self, *args): self.buffer.append(self.Transition(*args))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory = ReplayBuffer(Config.BUFFER_SIZE)
        self.steps = 0
        self.action_dim = action_dim
        
    def select_action(self, state, training=True):
        eps = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * \
              math.exp(-1. * self.steps / Config.EPSILON_DECAY)
        self.steps += 1
        if training and random.random() < eps: return random.randrange(self.action_dim)
        with torch.no_grad(): return self.policy_net(torch.FloatTensor(state)).argmax().item()
                
    def update(self):
        if len(self.memory) < Config.BATCH_SIZE: return
        batch = self.memory.Transition(*zip(*self.memory.sample(Config.BATCH_SIZE)))
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q = reward_batch + (Config.GAMMA * next_q * (1 - done_batch))
        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target(self): self.target_net.load_state_dict(self.policy_net.state_dict())

class LocalSearch:
    def optimize(self, machines, r_iii):
        optimized_machines = copy.deepcopy(machines)
        for m in optimized_machines:
            maint_indices = [i for i, x in enumerate(m.history) if 'Maint' in x[4]]
            for idx in reversed(maint_indices):
                event = m.history[idx]
                original_history = copy.deepcopy(m.history)
                duration = event[3] - event[2]
                del m.history[idx]
                valid_removal = True
                temp_age = 0
                for i, h in enumerate(m.history):
                    if i >= idx:
                        m.history[i] = (h[0], h[1], h[2]-duration, h[3]-duration, h[4])
                    h_type = m.history[i][4]
                    if 'JOB' in h_type:
                        rel = math.exp(-(temp_age/m.eta)**m.beta)
                        if rel <= r_iii:
                            valid_removal = False
                            break
                        temp_age += (m.history[i][3] - m.history[i][2])
                    elif 'Maint' in h_type:
                        f = Config.FACTOR_U if 'U' in h_type else (Config.FACTOR_V if 'V' in h_type else Config.FACTOR_W)
                        temp_age = temp_age * (1.0 - f)
                if not valid_removal: m.history = original_history
        return optimized_machines

class HeuristicAgent:
    def __init__(self, rule): self.rule = rule 
    def select_action(self, env):
        unfinished = [j for j in env.jobs if not j.finished]
        if not unfinished: return None
        if self.rule == 'FIFO': job = min(unfinished, key=lambda j: j.id)
        elif self.rule == 'SPT': job = min(unfinished, key=lambda j: j.ops_times[j.current_op_idx])
        elif self.rule == 'LPT': job = max(unfinished, key=lambda j: j.ops_times[j.current_op_idx])
        elif self.rule == 'MRT': job = max(unfinished, key=lambda j: j.get_remaining_proc_time())
        elif self.rule == 'RR': job = random.choice(unfinished)
        machine = min(env.machines, key=lambda m: m.available_time)
        return job, machine

# ==========================================
# 5. Worker Functions for Multiprocessing
# ==========================================
def train_dqn_worker(num_jobs, num_machines, q_limit, r_iii):
    torch.set_num_threads(1)
    env = DFJSP_Env(num_jobs, num_machines, q_limit, r_iii)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    for e in range(Config.TRAIN_EPISODES):
        state, _ = env.reset(seed=Config.SEED + e)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, r, done, _, _ = env.step(action)
            agent.memory.push(state, action, r, next_state, done)
            agent.update()
            state = next_state
        if e % Config.TARGET_UPDATE == 0: agent.update_target()
    return agent

def run_heuristic_episode(env, agent):
    state, _ = env.reset()
    done = False
    while not done:
        job, machine = agent.select_action(env)
        r_curr = machine.get_reliability()
        if r_curr <= env.r_iii:
            m_start = machine.available_time
            real_start, real_end = env.stw.find_slot(m_start, Config.DUR_W)
            machine.available_time = real_end
            machine.age = machine.age * (1.0 - Config.FACTOR_W)
            machine.history.append((-1, -1, real_start, real_end, 'Maint_W'))
        
        nominal = job.ops_times[job.current_op_idx]
        actual = machine.calculate_actual_time(nominal, Config.R_II)
        start = max(machine.available_time, job.next_ready_time)
        end = start + actual
        
        machine.available_time = end
        machine.age += actual
        machine.history.append((job.id, job.current_op_idx, start, end, 'JOB'))
        
        job.next_ready_time = end
        job.current_op_idx += 1
        if job.current_op_idx >= job.num_ops:
            job.finished = True
            env.finished_jobs += 1
        done = (env.finished_jobs == env.num_jobs)
    return max([m.available_time for m in env.machines])

# --- Worker for Table 6 (Comparative) ---
def worker_comparative(params):
    scale, case_idx, case_seed = params
    jobs, machines = scale
    dqn_agent = train_dqn_worker(jobs, machines, Config.DEFAULT_Q, Config.DEFAULT_R_III)
    
    row = {'Scale': f'{jobs}x{machines}', 'Case': case_idx + 1}
    
    # 1. DQN-LS
    dqn_vals = []
    ls = LocalSearch()
    for r in range(Config.REPLICAS):
        env = DFJSP_Env(jobs, machines, Config.DEFAULT_Q, Config.DEFAULT_R_III)
        state, _ = env.reset(seed=case_seed + r*1000)
        done = False
        while not done:
            action = dqn_agent.select_action(state, training=False)
            state, _, done, _, _ = env.step(action)
        opt_machines = ls.optimize(env.machines, Config.DEFAULT_R_III)
        dqn_vals.append(max([m.available_time for m in opt_machines]))
    row['DQN-LS Mean'] = np.mean(dqn_vals)
    row['DQN-LS Std'] = np.std(dqn_vals)
    
    # 2. Heuristics
    for rule in ['FIFO', 'LPT', 'SPT', 'MRT', 'RR']:
        h_vals = []
        for r in range(Config.REPLICAS):
            env = DFJSP_Env(jobs, machines, Config.DEFAULT_Q, Config.DEFAULT_R_III)
            env.reset(seed=case_seed + r*1000)
            agent = HeuristicAgent(rule)
            h_vals.append(run_heuristic_episode(env, agent))
        row[f'{rule} Mean'] = np.mean(h_vals)
        row[f'{rule} Std'] = np.std(h_vals)
        
    return row

# --- Worker for Table 4 (R_III) ---
def worker_sensitivity_r(params):
    scale, case_idx, case_seed, r_val = params
    jobs, machines = scale
    
    agent = train_dqn_worker(jobs, machines, Config.DEFAULT_Q, r_val)
    vals = []
    ls = LocalSearch()
    for rep in range(Config.REPLICAS):
        env = DFJSP_Env(jobs, machines, Config.DEFAULT_Q, r_val)
        state, _ = env.reset(seed=case_seed + rep*1000)
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, _, _ = env.step(action)
        opt = ls.optimize(env.machines, r_val)
        vals.append(max([m.available_time for m in opt]))
        
    return {
        'Scale': f'{jobs}x{machines}',
        'Case': case_idx + 1,
        'R_III': r_val,
        'Mean': np.mean(vals),
        'Std': np.std(vals)
    }

# --- Worker for Table 5 (Q) ---
def worker_sensitivity_q(params):
    scale, case_idx, case_seed, q_val = params
    jobs, machines = scale
    
    agent = train_dqn_worker(jobs, machines, q_val, Config.DEFAULT_R_III)
    vals = []
    ls = LocalSearch()
    for rep in range(Config.REPLICAS):
        env = DFJSP_Env(jobs, machines, q_val, Config.DEFAULT_R_III)
        state, _ = env.reset(seed=case_seed + rep*1000)
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, _, _ = env.step(action)
        opt = ls.optimize(env.machines, Config.DEFAULT_R_III)
        vals.append(max([m.available_time for m in opt]))
        
    return {
        'Scale': f'{jobs}x{machines}',
        'Case': case_idx + 1,
        'Q': q_val if q_val < 99 else 'Unlimited',
        'Mean': np.mean(vals),
        'Std': np.std(vals)
    }

# ==========================================
# 6. Main Execution with Multiprocessing
# ==========================================
def run_parallel_experiments():
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} cores. Starting parallel execution...")
    
    # --- 1. Comparative Experiment (Table 6) ---
    print("\n>>> Starting Comparative Experiment (Table 6)...")
    tasks_comp = []
    for scale in Config.SCALES:
        for idx, seed in enumerate(Config.CASE_SEEDS):
            tasks_comp.append((scale, idx, seed))
            
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_comp = list(tqdm(pool.imap(worker_comparative, tasks_comp), total=len(tasks_comp)))
        
    df_comp = pd.DataFrame(results_comp)
    df_comp.to_csv("table6_comparison.csv", index=False)
    print("Saved table6_comparison.csv")

    # --- 2. Sensitivity Analysis R (Table 4) ---
    print("\n>>> Starting Sensitivity Analysis R (Table 4)...")
    r_values = [0.76, 0.78, 0.80, 0.82, 0.84, 0.86]
    tasks_r = []
    for scale in Config.SCALES:
        for idx, seed in enumerate(Config.CASE_SEEDS):
            for r in r_values:
                tasks_r.append((scale, idx, seed, r))
                
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_r = list(tqdm(pool.imap(worker_sensitivity_r, tasks_r), total=len(tasks_r)))
        
    df_r_raw = pd.DataFrame(results_r)
    df_r_raw.to_csv("table4_r_sensitivity_raw.csv", index=False)
    
    df_r_wide = df_r_raw.pivot(index=['Scale', 'Case'], columns='R_III', values=['Mean', 'Std'])
    df_r_wide.columns = [f'R={c[1]} {c[0]}' for c in df_r_wide.columns]
    df_r_wide.reset_index().to_csv("table4_r_sensitivity.csv", index=False)
    print("Saved table4_r_sensitivity.csv")

    # --- 3. Sensitivity Analysis Q (Table 5) ---
    print("\n>>> Starting Sensitivity Analysis Q (Table 5)...")
    q_values = [1, 2, 3, 4, 99]
    tasks_q = []
    for scale in Config.SCALES:
        for idx, seed in enumerate(Config.CASE_SEEDS):
            for q in q_values:
                tasks_q.append((scale, idx, seed, q))
                
    with multiprocessing.Pool(processes=num_cores) as pool:
        results_q = list(tqdm(pool.imap(worker_sensitivity_q, tasks_q), total=len(tasks_q)))
        
    df_q_raw = pd.DataFrame(results_q)
    df_q_raw.to_csv("table5_q_sensitivity_raw.csv", index=False)
    
    df_q_wide = df_q_raw.pivot(index=['Scale', 'Case'], columns='Q', values=['Mean', 'Std'])
    df_q_wide.columns = [f'Q={c[1]} {c[0]}' for c in df_q_wide.columns]
    df_q_wide.reset_index().to_csv("table5_q_sensitivity.csv", index=False)
    print("Saved table5_q_sensitivity.csv")
    
    return df_r_raw, df_q_raw

# ==========================================
# 7. Plotting (REVISED for Exact Paper Match)
# ==========================================
def plot_results(df_r, df_q):
    # --- Figure 5: R_III Analysis (Dual Axis) ---
    target_scale = '15x8'
    target_case = 1
    
    subset = df_r[(df_r['Scale'] == target_scale) & (df_r['Case'] == target_case)].copy()
    
    if not subset.empty:
        subset = subset.sort_values('R_III')
        r_vals = subset['R_III'].values
        means = subset['Mean'].values
        stds = subset['Std'].values
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Dynamic Y-Limits for Makespan
        y_min = min(means) * 0.95
        y_max = max(means) * 1.05
        ax1.set_ylim(y_min, y_max)
        
        # Left Axis: Average Makespan (Blue Bars)
        width = 0.01
        bars = ax1.bar(r_vals, means, width=width, color='#4F94CD', label='Average makespan')
        ax1.set_xlabel('R_III Value')
        ax1.set_ylabel('Cmax (Makespan)', fontsize=12)
        ax1.set_xticks(r_vals)
        
        for rect in bars:
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., height + (y_max-y_min)*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        # Right Axis: Standard Deviation (Yellow Line)
        ax2 = ax1.twinx()
        ax2.plot(r_vals, stds, color='#EEC900', linestyle=':', marker='^', label='Standard deviation')
        ax2.set_ylabel('Std', fontsize=12, rotation=0, labelpad=15)
        
        std_max = max(stds) * 1.3
        ax2.set_ylim(0, std_max)
        
        for i, txt in enumerate(stds):
            ax2.text(r_vals[i], txt + std_max*0.03, f'{txt:.2f}', color='#EEC900', ha='center', fontsize=9)

        # Average Std Line
        avg_std = np.mean(stds)
        ax2.axhline(avg_std, color='#CD3333', linestyle='--', linewidth=2, label='Average standard deviation')
        ax2.text(r_vals[-1], avg_std, f'{avg_std:.2f}', color='#CD3333', va='center', ha='left', fontsize=10)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        
        plt.title(f"Figure 5: The parameter R_III analysis for the '{target_scale}, Q=3, case {target_case}'", y=-0.15)
        plt.tight_layout()
        plt.savefig("figure5_reproduced.png")
        plt.close()
        print("Saved figure5_reproduced.png")

    # --- Figure 6: Q Analysis (3 Subplots) ---
    scales = ['6x6', '15x8', '20x10']
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    for i, scale in enumerate(scales):
        ax = axes[i]
        subset_q = df_q[(df_q['Scale'] == scale) & (df_q['Case'] == 1)].copy()
        
        if subset_q.empty: continue
        
        unlimited_row = subset_q[subset_q['Q'] == 'Unlimited']
        unlimited_val = unlimited_row['Mean'].values[0] if not unlimited_row.empty else 0
        
        bars_data = subset_q[subset_q['Q'] != 'Unlimited'].copy()
        bars_data['Q'] = bars_data['Q'].astype(int)
        bars_data = bars_data.sort_values('Q')
        
        qs = bars_data['Q'].values
        means = bars_data['Mean'].values
        
        bars = ax.bar(qs.astype(str), means, color='#4F94CD', width=0.5, label='Average makespan')
        
        ax.axhline(unlimited_val, color='#CD3333', linestyle='--', linewidth=1.5, label='Unlimited Q')
        ax.text(3.5, unlimited_val, f'{unlimited_val:.2f}', color='#CD3333', va='bottom', ha='right', fontsize=9)
        
        for rect, val in zip(bars, means):
            height = rect.get_height()
            percent = ((val - unlimited_val) / unlimited_val) * 100
            ax.text(rect.get_x() + rect.get_width()/2., height + max(means)*0.01,
                    f'{height:.2f}\n+{percent:.2f}%', ha='center', va='bottom', fontsize=8)
            
        ax.set_ylim(0, max(means) * 1.3)
        ax.set_ylabel('Cmax')
        ax.set_xlabel(f"({chr(97+i)}) The limited maintenance resources analysis for the \"{scale} R_III=0.80 case 1\"")
        
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig("figure6_reproduced.png")
    plt.close()
    print("Saved figure6_reproduced.png")

# ==========================================
# 8. Demo Charts (Fig 7 & 8) - REVISED
# ==========================================
def get_job_color(job_id):
    # Consistent color map for up to 20 jobs
    cmap = plt.cm.get_cmap('tab20', 20)
    return cmap(job_id % 20)

def run_demo_experiment():
    print("\n>>> Generating Demo Charts (Fig 7 & 8)...")
    
    # --- Figure 7: Gantt Charts (2 Subplots) ---
    fig7, axes7 = plt.subplots(2, 1, figsize=(12, 10))
    
    # Case 1: 6x6
    print("Running 6x6 Case 1...")
    agent_6 = train_dqn_worker(6, 6, 3, 0.80)
    env_6 = DFJSP_Env(6, 6, 3, 0.80)
    state, _ = env_6.reset(seed=Config.CASE_SEEDS[0]) # Case 1 Seed
    done = False
    while not done:
        action = agent_6.select_action(state, training=False)
        state, _, done, _, _ = env_6.step(action)
    ls = LocalSearch()
    opt_machines_6 = ls.optimize(env_6.machines, 0.80)
    
    # Plot 6x6 on axes7[0]
    for i, m in enumerate(opt_machines_6):
        for h in m.history:
            start, end, type_ = h[2], h[3], h[4]
            dur = end - start
            if 'JOB' in type_:
                job_id = h[0]
                axes7[0].add_patch(mpatches.Rectangle((start, i*10), dur, 9, facecolor=get_job_color(job_id), edgecolor='black'))
                axes7[0].text(start+dur/2, i*10+4.5, f"{job_id+1}-{h[1]+1}", ha='center', va='center', color='white', fontsize=8)
            else:
                axes7[0].add_patch(mpatches.Rectangle((start, i*10), dur, 9, facecolor='brown', hatch='//', edgecolor='black'))
                axes7[0].text(start+dur/2, i*10+4.5, type_[-1], ha='center', va='center', fontsize=8, color='white')
    axes7[0].set_yticks([i*10+5 for i in range(6)])
    axes7[0].set_yticklabels([f'Machine{i+1}' for i in range(6)])
    axes7[0].set_title('(a) Gantt Chart for "6 x 6 Q = 3 R_III = 0.80 case 1"')
    axes7[0].set_xlabel("Time")
    axes7[0].autoscale()

    # Case 2: 20x10
    print("Running 20x10 Case 1...")
    agent_20 = train_dqn_worker(20, 10, 3, 0.80)
    env_20 = DFJSP_Env(20, 10, 3, 0.80)
    state, _ = env_20.reset(seed=Config.CASE_SEEDS[0]) # Case 1 Seed
    done = False
    while not done:
        action = agent_20.select_action(state, training=False)
        state, _, done, _, _ = env_20.step(action)
    opt_machines_20 = ls.optimize(env_20.machines, 0.80)
    
    # Plot 20x10 on axes7[1]
    for i, m in enumerate(opt_machines_20):
        for h in m.history:
            start, end, type_ = h[2], h[3], h[4]
            dur = end - start
            if 'JOB' in type_:
                job_id = h[0]
                axes7[1].add_patch(mpatches.Rectangle((start, i*10), dur, 9, facecolor=get_job_color(job_id), edgecolor='black'))
                # Text might be too small for 20x10, optional
                if dur > 2:
                    axes7[1].text(start+dur/2, i*10+4.5, f"{job_id+1}", ha='center', va='center', color='white', fontsize=6)
            else:
                axes7[1].add_patch(mpatches.Rectangle((start, i*10), dur, 9, facecolor='brown', hatch='//', edgecolor='black'))
    axes7[1].set_yticks([i*10+5 for i in range(10)])
    axes7[1].set_yticklabels([f'Machine{i+1}' for i in range(10)])
    axes7[1].set_title('(b) Gantt Chart for "20 x 10 Q = 3 R_III = 0.80 case 1"')
    axes7[1].set_xlabel("Time")
    axes7[1].autoscale()
    
    plt.tight_layout()
    plt.savefig("figure7_reproduced.png")
    plt.close()
    print("Saved figure7_reproduced.png")

    # --- Figure 8: Trajectory Plots (6 Subplots) ---
    # Using opt_machines_6 (6x6 Case 1)
    fig8, axes8 = plt.subplots(3, 2, figsize=(12, 12))
    axes8 = axes8.flatten()
    
    for m_idx in range(6):
        ax = axes8[m_idx]
        m = opt_machines_6[m_idx]
        
        # Reconstruct trajectory points
        times = [0]
        ages = [0]
        current_age = 0
        sorted_h = sorted(m.history, key=lambda x: x[2])
        
        # Plot segments
        for h in sorted_h:
            start, end, type_ = h[2], h[3], h[4]
            
            # Idle time (age constant)
            if start > times[-1]:
                ax.plot([times[-1], start], [current_age, current_age], color='black', linestyle=':')
            
            # Action
            prev_age = current_age
            if 'JOB' in type_:
                current_age += (end - start)
                color = get_job_color(h[0]) # Match Gantt color
                ax.plot([start, end], [prev_age, current_age], color=color, marker='.', linewidth=1.5)
                # Annotate job
                ax.text((start+end)/2, (prev_age+current_age)/2 + 0.5, f"({start:.0f},{prev_age:.0f})", fontsize=6)
            else:
                f = Config.FACTOR_U if 'U' in type_ else (Config.FACTOR_V if 'V' in type_ else Config.FACTOR_W)
                current_age *= (1.0 - f)
                ax.plot([start, end], [prev_age, current_age], color='brown', linestyle='--', marker='x')
            
            times.append(end)
            ages.append(current_age)
            
        # Thresholds
        age_II = m.eta * ((-math.log(Config.R_II)) ** (1/m.beta))
        age_III = m.eta * ((-math.log(Config.DEFAULT_R_III)) ** (1/m.beta))
        
        ax.axhline(y=age_II, color='orange', linestyle='-', linewidth=1, label='R_II')
        ax.axhline(y=age_III, color='red', linestyle='-', linewidth=1, label='R_III')
        
        ax.set_title(f"({chr(97+m_idx)}) The trajectory plot for Machine {m_idx+1} age")
        ax.set_xlabel("Time")
        ax.set_ylabel("Effective Age")
        ax.grid(True, alpha=0.3)
        
        if m_idx == 1: # Legend on one plot
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig("figure8_reproduced.png")
    plt.close()
    print("Saved figure8_reproduced.png")

if __name__ == "__main__":
    # 1. Run Parallel Experiments
    df_r, df_q = run_parallel_experiments()
    
    # 2. Plot Results
    plot_results(df_r, df_q)
    
    # 3. Run Demo
    run_demo_experiment()
    
    print("\nAll tasks completed successfully.")