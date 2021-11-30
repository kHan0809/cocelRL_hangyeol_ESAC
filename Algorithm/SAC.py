import torch
import torch.nn.functional as F
import numpy as np
from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update

from Model.Model import Critic, Squashed_Gaussian_Actor, Mixture_Squashed_Gaussian_Actor

class SAC_fix_alpha:
    def __init__(self,state_dim,action_dim,device,args):
        self.device = device
        self.buffer_size = args.buffer_size
        self.batch_size  = args.batch_size


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = args.tau
        self.gamma = args.gamma
        self.current_step = 0
        self.training_start = args.training_start
        self.training_step = args.training_step
        self.critic_update = args.critic_update
        self.train_alpha_flag = args.train_alpha_flag
        self.target_entropy = -action_dim
        self.log_alpha = torch.as_tensor(np.log(args.alpha), dtype=torch.float32, device=self.device).requires_grad_()
        # self.log_beta = torch.as_tensor(np.log(args.beta), dtype=torch.float32, device=self.device).requires_grad_()

        self.buffer = Buffer(self.state_dim, self.action_dim, self.buffer_size, self.device)

        self.actor = Mixture_Squashed_Gaussian_Actor(self.state_dim, self.action_dim, args.log_std_min, args.log_std_max).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)


        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @property
    def beta(self):
        return self.log_beta.exp().detach()

    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _, _ = self.actor(state)

            action = np.clip(action.cpu().numpy()[0], -1, 1)

        return action

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _, _ = self.actor(state, deterministic=True)

            action = np.clip(action.cpu().numpy()[0], -1, 1)

        return action

    def train_alpha(self, s):
        _, s_logpi, _ = self.actor(s)
        alpha_loss = -(self.log_alpha.exp() * (s_logpi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def train_critic(self, s, a, r, ns, d):
        ns_action, ns_logpi_ttl, ns_logpi_eps = self.actor(ns)

        target_min_aq = torch.minimum(self.target_critic1(ns, ns_action), self.target_critic2(ns, ns_action))
        target_q = (r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logpi_ttl)).detach()
        # target_q = (r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logpi_ttl + self.beta * ns_logpi_eps)).detach()

        critic1_loss = F.mse_loss(input=self.critic1(s, a), target=target_q)
        critic2_loss = F.mse_loss(input=self.critic2(s, a), target=target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return (critic1_loss.item(), critic2_loss.item())

    def train_actor(self, s):
        s_action, s_logpi_ttl, s_logpi_eps = self.actor(s)
        min_aq_rep = torch.minimum(self.critic1(s, s_action), self.critic2(s, s_action))
        # actor_loss = (self.alpha * s_logpi_ttl - self.beta * s_logpi_eps - min_aq_rep).mean()
        actor_loss = (self.alpha * s_logpi_ttl - min_aq_rep).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def train(self, training_num):
        total_a_loss = 0
        total_c1_loss, total_c2_loss = 0, 0
        total_alpha_loss = 0
        for i in range(training_num):
            self.current_step += 1
            s, a, r, ns, d = self.buffer.get_batch(self.batch_size)


            critic1_loss, critic2_loss = self.train_critic(s, a, r, ns, d)
            total_c1_loss += critic1_loss
            total_c2_loss += critic2_loss
            total_a_loss += self.train_actor(s)

            if self.train_alpha_flag  == True:
                total_alpha_loss += self.train_alpha(s)

            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)

        return [['Loss/Actor', total_a_loss], ['Loss/Critic1', total_c1_loss], ['Loss/Critic2', total_c2_loss],
                ['Loss/alpha', total_alpha_loss], ['Alpha', self.alpha]]

