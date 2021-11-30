import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixture_Squashed_Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-10, log_std_max=2,n_mixtures = 5):
        super(Mixture_Squashed_Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_mixture = n_mixtures

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max


        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_dim*self.n_mixture*3)


    def forward(self, state, deterministic = False):
        L1 = F.relu(self.fc1(state))
        L2 = F.relu(self.fc2(L1))
        output = torch.tanh(self.fc3(L2))

        mix_weight, mix_mean, log_mix_std = output.chunk(3, dim=-1)
        mix_weight     = mix_weight.reshape(-1, self.action_dim, self.n_mixture)
        mix_mean       = mix_mean.reshape(-1, self.action_dim, self.n_mixture)
        log_mix_std    = log_mix_std.reshape(-1, self.action_dim, self.n_mixture)

        max_weight = torch.max(mix_weight,dim=2).values
        max_weight = torch.unsqueeze(max_weight,2)
        nrd_mix_weight = F.softmax(mix_weight-max_weight,dim=2)
        log_std = torch.clamp(log_mix_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        mean = torch.sum(nrd_mix_weight * mix_mean,dim=2)
        alea = torch.sum(nrd_mix_weight * std,dim=2)
        epis = torch.sum(nrd_mix_weight*torch.square(mix_mean-torch.unsqueeze(mean,dim=2)),dim=2)
        epis = torch.clamp(epis, 4.53999297624848e-05, 7.38905609893065)
        total = alea + epis

        dist_ttl = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(total.pow(2),offset=0,dim1=-2,dim2=-1))
        dist_eps = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(epis.pow(2),offset=0,dim1=-2,dim2=-1))

        if deterministic == True:
            tanh_mean = torch.tanh(mean)
            log_prob_ttl = dist_ttl.log_prob(mean)
            log_prob_eps = dist_eps.log_prob(mean)

            log_pi_ttl = log_prob_ttl.view(-1,1) - torch.log(1 - tanh_mean.pow(2) + 1e-6).sum(dim=1,keepdim=True)
            log_pi_eps = log_prob_eps.view(-1, 1) - torch.log(1 - tanh_mean.pow(2) + 1e-6).sum(dim=1, keepdim=True)
            return tanh_mean, log_pi_ttl, log_pi_eps

        else:
            sample_action = dist_ttl.rsample()
            tanh_sample = torch.tanh(sample_action)
            log_prob_ttl = dist_ttl.log_prob(sample_action)
            log_prob_eps = dist_eps.log_prob(sample_action)

            log_pi_ttl = log_prob_ttl.view(-1,1) - torch.log(1 - tanh_sample.pow(2) + 1e-6).sum(dim=1,keepdim=True)
            log_pi_eps = log_prob_eps.view(-1, 1) - torch.log(1 - tanh_sample.pow(2) + 1e-6).sum(dim=1, keepdim=True)
            return tanh_sample, log_pi_ttl, log_pi_eps

class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-10, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_dim*2)


    def forward(self, state, deterministic = False):
        L1 = F.relu(self.fc1(state))
        L2 = F.relu(self.fc2(L1))
        output = torch.tanh(self.fc3(L2))

        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(std.pow(2),offset=0,dim1=-2,dim2=-1))

        if deterministic == True:
            tanh_mean = torch.tanh(mean)
            log_prob = dist.log_prob(mean)

            log_pi = log_prob.view(-1,1) - torch.log(1 - tanh_mean.pow(2) + 1e-6).sum(dim=1,keepdim=True)
            return tanh_mean, log_pi

        else:
            sample_action = dist.rsample()
            tanh_sample = torch.tanh(sample_action)
            log_prob = dist.log_prob(sample_action)

            log_pi = log_prob.view(-1,1) - torch.log(1 - tanh_sample.pow(2) + 1e-6).sum(dim=1,keepdim=True)
            return tanh_sample, log_pi


class Actor(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.action_dim)

    def forward(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        output = torch.tanh(self.fc3(L2))
        return output


class Critic(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim+self.action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, s,a):
        x = torch.cat((s, a), 1)
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        output = self.fc3(L2)

        return output


class V_net(nn.Module):
    def __init__(self,state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.fc1 = nn.Linear(self.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        output = self.fc3(L2)

        return output
