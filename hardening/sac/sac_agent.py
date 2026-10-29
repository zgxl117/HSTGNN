# sac/sac_agent.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import os
import time

class SACAgent:
    """SAC代理"""
    
    def __init__(self, state_dim, action_dim, action_space, config, device=None):
        # 自动检测并选择NVIDIA GPU
        if device is None:
            if torch.cuda.is_available():
                # 检查是否有多个GPU，优先选择NVIDIA
                device_count = torch.cuda.device_count()
                selected_device = 0

                # 尝试查找NVIDIA GPU
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    if 'nvidia' in gpu_name.lower() or 'geforce' in gpu_name.lower():
                        selected_device = i
                        # print(f"找到NVIDIA GPU: {gpu_name}，选择设备{selected_device}")
                        break
                    
                device = f'cuda:{selected_device}'
                torch.cuda.set_device(selected_device)  # 设置默认设备

                # 配置CUDA优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

                # 验证设备
                current_device = torch.cuda.current_device()
                # print(f"当前CUDA设备: {current_device} - {torch.cuda.get_device_name(current_device)}")
            else:
                device = 'cpu'
                print("INFO: 使用CPU，未检测到可用GPU")
        else:
            device = device

        self.device = torch.device(device)
        print(f"最终使用设备: {self.device}")

        # 如果是CUDA设备，检查GPU内存
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            cached = torch.cuda.memory_reserved(self.device) / 1024**3
            # print(f"GPU内存: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space

        # 网络参数 - 根据设备调整
        if self.device.type == 'cuda':
            self.hidden_dim = config.get('hidden_dim', 256)  # GPU可以处理更大的网络
        else:
            self.hidden_dim = config.get('hidden_dim', 128)  # CPU使用较小的网络

        # SAC参数
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2)
        self.automatic_entropy_tuning = config.get('automatic_entropy_tuning', True)
        self.lr = config.get('lr', 3e-4)
        self.eval_noise_scale = float(config.get('eval_noise_scale', 0.01))
        self.use_prioritized_replay = bool(config.get('use_prioritized_replay', True))
        self.priority_beta = float(config.get('priority_beta', 0.4))
        self.priority_beta_increment = float(config.get('priority_beta_increment', 0.002))

        # 动态调整batch_size
        if self.device.type == 'cuda':
            # 1660 Super有6GB显存，可以设置较大的batch_size
            self.batch_size = config.get('batch_size', 512)
        else:
            # CPU使用较小的batch_size
            self.batch_size = config.get('batch_size', 256)

        # 创建网络并移动到设备
        self.critic = CriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = ActorNetwork(state_dim, action_dim, self.hidden_dim, action_space).to(self.device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # 自动熵调整
        if self.automatic_entropy_tuning:
            action_shape_tensor = torch.Tensor(action_space.shape).to(self.device)
            self.target_entropy = -torch.prod(action_shape_tensor).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        # 回放缓冲区 - 保持在CPU上（避免GPU内存溢出）
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('buffer_size', 500000),  # 1660 Super显存有限，减少缓冲区大小
            state_dim=state_dim,
            action_dim=action_dim,
            use_prioritized=self.use_prioritized_replay,
            priority_alpha=float(config.get('priority_alpha', 0.65)),
            priority_epsilon=float(config.get('priority_epsilon', 1e-3)),
        )

        self.update_step = 0

        # 性能监控
        self.update_times = []
    
    def _scale_unit_action(self, unit_action):
        """Scale tanh policy output from [-1, 1] to the real action space."""
        return self.action_space.low + (self.action_space.high - self.action_space.low) * (unit_action + 1.0) / 2.0

    def _normalize_action_tensor(self, action):
        """Map real MOEA parameters to [-1, 1] before critic evaluation."""
        low = torch.as_tensor(self.action_space.low, dtype=action.dtype, device=action.device)
        high = torch.as_tensor(self.action_space.high, dtype=action.dtype, device=action.device)
        span = torch.clamp(high - low, min=torch.finfo(action.dtype).eps)
        return torch.clamp(2.0 * (action - low) / span - 1.0, -1.0, 1.0)

    def select_action(self, state, evaluate=False, noise_scale=None):
        """Select an action.

        In stochastic mode SAC samples from the policy distribution. In evaluate
        mode it uses the policy mean, correctly scaled to the real action space.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            with torch.no_grad():
                _, _, mean_unit_action = self.actor.sample(state)
            action = self._scale_unit_action(mean_unit_action.cpu().numpy()[0])
            if noise_scale is None:
                noise_scale = self.eval_noise_scale
            if noise_scale and noise_scale > 0:
                action = action + np.random.randn(*action.shape) * noise_scale * (
                    self.action_space.high - self.action_space.low
                )
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)
            action = action.cpu().numpy()[0]

        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换到回放缓冲区"""
        priority = abs(float(reward)) + self.replay_buffer.priority_epsilon
        self.replay_buffer.push(
            state, action, reward, next_state, done, priority=priority
        )
    
    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 性能监控开始
        update_start_time = time.time()

        if self.device.type == 'cuda':
            start_allocated = torch.cuda.memory_allocated(self.device)
            start_reserved = torch.cuda.memory_reserved(self.device)
        
        # 采样批次（在CPU上）
        priority_beta = min(
            1.0,
            self.priority_beta + self.priority_beta_increment * self.update_step,
        )
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            replay_indices,
            sample_weights,
        ) = self.replay_buffer.sample(self.batch_size, beta=priority_beta)
        
        # 转换为张量并移动到设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        sample_weights = torch.FloatTensor(sample_weights).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_pis, _ = self.actor.sample(next_states)
            next_actions_normalized = self._normalize_action_tensor(next_actions)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions_normalized)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
        
        actions_normalized = self._normalize_action_tensor(actions)
        current_Q1, current_Q2 = self.critic(states, actions_normalized)
        td_error1 = current_Q1 - target_Q
        td_error2 = current_Q2 - target_Q
        critic_loss = (
            (sample_weights * td_error1.pow(2)).mean()
            + (sample_weights * td_error2.pow(2)).mean()
        )
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 记录梯度信息
        if self.device.type == 'cuda' and self.update_step % 100 == 0:
            grad_norm = 0
            for param in self.critic.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            print(f"Critic梯度范数: {grad_norm:.4f}")

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        with torch.no_grad():
            td_priority = 0.5 * (td_error1.abs() + td_error2.abs())
        self.replay_buffer.update_priorities(
            replay_indices,
            td_priority.squeeze(-1).detach().cpu().numpy(),
        )
        
        # 更新Actor
        actions_pred, log_pis, _ = self.actor.sample(states)
        actions_pred_normalized = self._normalize_action_tensor(actions_pred)
        Q1, Q2 = self.critic(states, actions_pred_normalized)
        Q = torch.min(Q1, Q2)
        
        actor_loss = (self.alpha * log_pis - Q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 自动调整温度参数
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.update_step += 1
        
        # 性能监控
        update_time = time.time() - update_start_time
        self.update_times.append(update_time)
        
        # 记录GPU内存变化
        if self.device.type == 'cuda':
            end_allocated = torch.cuda.memory_allocated(self.device)
            end_reserved = torch.cuda.memory_reserved(self.device)

            if self.update_step % 50 == 0:
                print(f"\n[GPU监控] 步数: {self.update_step}")
                print(f"  分配内存: {start_allocated/1024**3:.3f}GB → {end_allocated/1024**3:.3f}GB")
                print(f"  保留内存: {start_reserved/1024**3:.3f}GB → {end_reserved/1024**3:.3f}GB")
                print(f"  变化: +{(end_allocated - start_allocated)/1024**3:.3f}GB")

        # 定期清理GPU缓存
        if self.device.type == 'cuda' and self.update_step % 200 == 0:
            torch.cuda.empty_cache()
            # print(f"已清理GPU缓存 (步数: {self.update_step})")
        
        # 记录训练统计
        if self.update_step % 100 == 0:
            avg_update_time = np.mean(self.update_times[-100:]) if len(self.update_times) > 0 else 0
            print(f"\nDEBUG[SAC]: 更新步数={self.update_step}")
            print(f"  Critic损失={critic_loss.item():.6f}, Actor损失={actor_loss.item():.6f}")
            print(f"  Alpha={self.alpha.item():.4f}, 温度参数={self.log_alpha.item():.4f}")
            print(f"  平均更新时间={avg_update_time*1000:.2f}ms")
            # print(f"  设备={self.device}")
            
            # 显示网络参数统计
            total_params = sum(p.numel() for p in self.actor.parameters())
            trainable_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
            print(f"  网络参数: {total_params:,} (可训练: {trainable_params:,})")
    
    def check_gpu_memory(self):
        """检查GPU内存使用情况"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(self.device) / 1024**3  # GB

            ## print(f"GPU内存使用: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

            # 如果内存使用过高，进行清理
            if allocated > 4.0:  # 1660 Super有6GB显存，超过4GB时清理
                torch.cuda.empty_cache()
                ## print("INFO: GPU内存使用过高，已清理缓存")

    def save_checkpoint(self, filepath):
        """保存检查点"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_step': self.update_step,
        }
        
        if self.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"模型保存到: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        if not os.path.exists(filepath):
            print(f"检查点不存在: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.update_step = checkpoint['update_step']
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        print(f"从 {filepath} 加载模型")

class ActorNetwork(nn.Module):
    """Actor网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim, action_space):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_space = action_space
        
        # 网络架构
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            module.bias.data.zero_()
    
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化技巧
        y_t = torch.tanh(x_t)
        
        # 计算log概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的概率
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 将动作缩放到动作空间范围
        action_low = torch.tensor(self.action_space.low, device=state.device)
        action_high = torch.tensor(self.action_space.high, device=state.device)
        
        action = action_low + (action_high - action_low) * (y_t + 1) / 2
        
        return action, log_prob, torch.tanh(mean)

class CriticNetwork(nn.Module):
    """Critic网络（双Q网络）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2网络
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            module.bias.data.zero_()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        return q1, q2

class ReplayBuffer:
    """支持优先经验回放的CPU缓冲区。"""
    
    def __init__(
        self,
        capacity,
        state_dim,
        action_dim,
        use_prioritized=True,
        priority_alpha=0.65,
        priority_epsilon=1e-3,
    ):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.use_prioritized = bool(use_prioritized)
        self.priority_alpha = float(priority_alpha)
        self.priority_epsilon = float(priority_epsilon)
        
        # 初始化缓冲区
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """存储转换。"""
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        if priority is None:
            priority = (
                float(np.max(self.priorities[:self.size]))
                if self.size > 0
                else 1.0
            )
        self.priorities[idx] = max(
            abs(float(priority)), self.priority_epsilon
        )
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        """按优先级采样，并返回重要性采样权重。"""
        if self.use_prioritized:
            valid_priorities = np.maximum(
                self.priorities[:self.size], self.priority_epsilon
            )
            scaled = np.power(valid_priorities, self.priority_alpha)
            probabilities = scaled / max(float(np.sum(scaled)), self.priority_epsilon)
            indices = np.random.choice(
                self.size, size=batch_size, replace=True, p=probabilities
            )
            sample_weights = np.power(
                self.size * probabilities[indices], -float(beta)
            )
            sample_weights /= max(float(np.max(sample_weights)), self.priority_epsilon)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)
            sample_weights = np.ones(batch_size, dtype=np.float32)
        
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            indices,
            np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1),
        )

    def update_priorities(self, indices, priorities):
        """用最新TD误差更新样本优先级。"""
        if not self.use_prioritized:
            return
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = max(
                abs(float(priority)), self.priority_epsilon
            )
    
    def __len__(self):
        return self.size
