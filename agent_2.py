import torch
import random, numpy as np
from pathlib import Path

from neural_2 import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=32)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay =  0.999999# 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 0 #1e4  # min. experiences before training
        self.learn_every = 32   # no. of experiences between updates to Q_online
        self.sync_every = 0  #1e3   # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e5   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00001)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state, deterministic=False):
        state = np.array(state)
        action, log_prob = self.net.act(state, deterministic)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action, log_prob

    def cache(self, state, next_state, action, reward, done, log_prob):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done, log_prob) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done, log_prob = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), log_prob.squeeze()


    def learn(self):
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, rewards, done, log_probs = self.recall()

        returns = deque(maxlen=self.batch_size)

        for t in range(self.batch_size)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(self.gamma * disc_return_t + rewards[t])

        ## standardization of the returns to make training more stable
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Compute loss
        policy_loss = []
        for log_prob, disc_return in zip(log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss =  torch.stack(policy_loss)
        policy_loss = policy_loss.sum()

        # Policy update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return (0, policy_loss.item())


    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = 0 #exploration_rate
