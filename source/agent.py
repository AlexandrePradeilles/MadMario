import torch
import torch.nn as nn
import torch.optim as optim
import random, numpy as np
from pathlib import Path

from neural import feat_extractor, RNDModel, QNetwork
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.actions_mem = deque(maxlen=10000)
        self.batch_size = 32

        # define the exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 300000 #3000000
        self.epsilon_by_step = lambda step: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1.0 * step / self.epsilon_decay)
        
        # define the training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.001
        self.extrinsec_w = 0.66

        self.curr_step = 0
        self.burnin = 1e2  # min. experiences before training
        self.learn_every = 4   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 1e5   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.feature_extractor = feat_extractor(4)
        self.target_feature_extractor = feat_extractor(4)

        # define the RND optimizer
        self.rnd_model = RNDModel(self.feature_extractor)
        self.rnd_target_model = RNDModel(self.target_feature_extractor)
        if self.use_cuda:
            self.rnd_model = self.rnd_model.to(device='cuda')
            self.rnd_target_model = self.rnd_target_model.to(device='cuda')
        self.rnd_optimizer = optim.Adam(self.rnd_model.predictor.parameters(), lr=0.00001)

        # define the target network
        self.target_network = QNetwork(state_dim, action_dim)
        if self.use_cuda:
            self.target_network = self.target_network.to(device='cuda')
        self.optimizer = optim.Adam(self.target_network.parameters(), lr=0.00001)
        self.mse_loss = nn.MSELoss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # decrease exploration_rate
        self.exploration_rate = self.epsilon_by_step(self.curr_step)
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
                state = state.unsqueeze(0)
                action_idx = torch.argmax(self.target_network(state)).item()
                self.actions_mem.append(action_idx)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
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

        self.memory.append( (state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def learn(self):
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None, None

        if self.curr_step % self.learn_every != 0:
            return None, None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # compute the Q target values
        with torch.no_grad():
            next_q_values = self.target_network(next_state)
            next_actions = torch.argmax(next_q_values, axis=1)
            rnd_int = self.rnd_target_model(next_state).clamp(-10.0,10.0)
            next_q_values = rnd_int + self.gamma * next_q_values[:, next_actions] * (1 - done.float())
            q_targets = (reward + self.extrinsec_w * next_q_values).float()

        # compute the Q predicted values
        q_values = self.target_network(state)[:, action].float()

        # update the target network
        target_network_params = self.target_network.named_parameters()
        target_feature_extractor_params = self.target_feature_extractor.named_parameters()
        target_params = dict(target_network_params)
        target_feature_extractor_params = dict(target_feature_extractor_params)
        for name, param in self.feature_extractor.named_parameters():
            target_feature_extractor_params[name].data.copy_(self.tau * param.data + (1 - self.tau) * target_feature_extractor_params[name].data)
        for name, param in self.target_network.named_parameters():
            if name in target_params:
                target_params[name].data.copy_(self.tau * param.data + (1 - self.tau) * target_params[name].data)

        # update the Q network
        self.optimizer.zero_grad()
        loss = self.mse_loss(q_values, q_targets.detach()).float()
        loss.backward()
        self.optimizer.step()

        # update the RND model
        self.rnd_optimizer.zero_grad()
        feature_states = self.feature_extractor(state)
        rnd_targets = self.rnd_target_model.predictor(feature_states).detach()
        rnd_predictions = self.rnd_model.predictor(feature_states)
        rnd_loss = self.mse_loss(rnd_predictions, rnd_targets)
        rnd_loss.backward()
        self.rnd_optimizer.step()

        # update the target network
        if self.curr_step % self.sync_every == 0:
            self.target_feature_extractor.load_state_dict(self.feature_extractor.state_dict())

        return (q_values.mean().item(), rnd_loss.item(), loss.item())


    def save(self):
        save_path = self.save_dir / f"mario_rnd_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.rnd_model.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        save_path = self.save_dir / f"mario_ql_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.target_network.state_dict(),
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
