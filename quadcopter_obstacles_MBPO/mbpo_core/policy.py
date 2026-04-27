from abc import ABC, abstractmethod
import math

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete, Space
except ImportError:  # pragma: no cover - compatibility fallback
    import gym
    from gym.spaces import Box, Discrete, Space
import torch
from torch import distributions as td

from .torch_util import torchify, Module, device
from .squashed_gaussian import SquashedGaussian


class BasePolicy(ABC):
    @abstractmethod
    def act(self, states, eval): pass

    def act1(self, state, eval=False):
        return self.act(torch.unsqueeze(state, 0), eval)[0]


class UniformPolicy(BasePolicy):
    def __init__(self, env_or_action_space):
        if isinstance(env_or_action_space, gym.Env):
            action_space = env_or_action_space.action_space
        elif isinstance(env_or_action_space, gym.Space):
            action_space = env_or_action_space
        else:
            raise ValueError('Must pass env or action space')

        if isinstance(action_space, Box):
            self.low = torchify(action_space.low, to_device=True)
            self.high = torchify(action_space.high, to_device=True)
            self.shape = list(action_space.shape)
            self.discrete = False
        elif isinstance(action_space, Discrete):
            self.n = action_space.n
            self.discrete = True
        else:
            raise NotImplementedError(f'Unsupported action space: {action_space}')

    def act(self, states, eval):
        batch_size = len(states)
        if self.discrete:
            return torch.randint(self.n, size=(batch_size,), device=device)
        else:
            return self.low + torch.rand(batch_size, *self.shape, device=device) * (self.high - self.low)

    def prob(self, actions):
        batch_size = len(actions)
        if self.discrete:
            assert actions.dim() == 1
            p = 1./self.n
        else:
            assert actions.dim() == 2
            p = 1./torch.prod(self.high - self.low)
        return torch.full([batch_size], p, device=device)

    def log_prob(self, actions):
        return torch.log(self.prob(actions))


class GaussianNoisePolicy(BasePolicy):
    def __init__(self, env_or_action_space, std=0.35):
        if isinstance(env_or_action_space, gym.Env):
            action_space = env_or_action_space.action_space
        elif isinstance(env_or_action_space, gym.Space):
            action_space = env_or_action_space
        else:
            raise ValueError('Must pass env or action space')

        if isinstance(action_space, Box):
            self.low = torchify(action_space.low, to_device=True)
            self.high = torchify(action_space.high, to_device=True)
            self.shape = list(action_space.shape)
            self.center = 0.5 * (self.low + self.high)
            self.half_range = 0.5 * (self.high - self.low)
            self.std = float(std)
            self.discrete = False
        elif isinstance(action_space, Discrete):
            self.n = action_space.n
            self.discrete = True
        else:
            raise NotImplementedError(f'Unsupported action space: {action_space}')

    def act(self, states, eval):
        batch_size = len(states)
        if self.discrete:
            return torch.randint(self.n, size=(batch_size,), device=device)
        noise = torch.randn(batch_size, *self.shape, device=device) * (self.std * self.half_range)
        return (self.center + noise).clamp(self.low, self.high)


class TorchPolicy(BasePolicy, Module):
    def __init__(self, net):
        Module.__init__(self)
        self.net = net
        self.use_special_eval = False

    @abstractmethod
    def _distr(self, *network_outputs): pass

    def distr(self, states):
        return self._distr(self.net(states))

    @abstractmethod
    def _special_eval(self, distr):
        raise NotImplementedError

    def act(self, states, eval):
        with torch.no_grad():
            distr = self.distr(states)
        if self.use_special_eval or eval:
            return self._special_eval(distr)
        if hasattr(distr, "rsample"):
            return distr.rsample()
        return distr.sample()


class SquashedGaussianPolicy(TorchPolicy):
    def __init__(self, net, log_std_bounds=(-5, 2), std_multiplier=1.0):
        super().__init__(net)
        self.log_std_bounds = log_std_bounds
        self.std_multiplier = std_multiplier

    def _mu_log_std(self, states):
        net_out = self.net(states)
        mu, log_std = net_out.chunk(2, dim=-1)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + (log_std_max - log_std_min) * torch.sigmoid(log_std)
        std = log_std.exp() * self.std_multiplier
        return mu, log_std, std

    def _distr(self, net_out):
        mu, log_std = net_out.chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + (log_std_max - log_std_min) * torch.sigmoid(log_std)
        # log_std = log_std.clamp(log_std_min, log_std_max)
        std = log_std.exp() * self.std_multiplier
        return td.Independent(SquashedGaussian(mu, std, validate_args=False), 1)

    def _special_eval(self, distr):
        return distr.mean

    def sample_and_log_prob(self, states):
        mu, log_std, std = self._mu_log_std(states)
        noise = torch.randn_like(mu)
        pre_tanh = mu + noise * std
        action = torch.tanh(pre_tanh)
        # Manual Gaussian log-prob avoids transformed-distribution overhead in Isaac GPU runs.
        gaussian_log_prob = -0.5 * (
            noise.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob = gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def act(self, states, eval):
        with torch.no_grad():
            mu, _, std = self._mu_log_std(states)
            if self.use_special_eval or eval:
                return torch.tanh(mu)
            noise = torch.randn_like(mu)
            return torch.tanh(mu + noise * std)
