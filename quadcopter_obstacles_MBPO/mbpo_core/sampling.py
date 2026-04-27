import h5py
import numpy as np
import torch

from .torch_util import device, Module, torchify, random_indices


class SampleBuffer(Module):
    """Generic replay buffer for vector observations and actions."""

    COMPONENT_NAMES = ("states", "actions", "next_states", "rewards", "dones")

    def __init__(self, state_dim, action_dim, capacity, discrete_actions=False, device=device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.discrete_actions = discrete_actions
        self.device = device

        self._bufs = {}
        self.register_buffer("_pointer", torch.tensor(0, dtype=torch.long))

        if discrete_actions:
            assert action_dim == 1
            action_dtype = torch.int
            action_shape = []
        else:
            action_dtype = torch.float
            action_shape = [action_dim]

        components = (
            ("states", torch.float, [state_dim]),
            ("actions", action_dtype, action_shape),
            ("next_states", torch.float, [state_dim]),
            ("rewards", torch.float, []),
            ("dones", torch.bool, []),
        )
        for name, dtype, shape in components:
            self._create_buffer(name, dtype, shape)

    @classmethod
    def from_state_dict(cls, state_dict, device=device):
        assert set(state_dict.keys()) == {*(f"_{name}" for name in cls.COMPONENT_NAMES), "_pointer"}
        states, actions = state_dict["_states"], state_dict["_actions"]

        length = len(states)
        for name in cls.COMPONENT_NAMES:
            tensor = state_dict[f"_{name}"]
            assert torch.is_tensor(tensor)
            assert len(tensor) == length

        action_dim = actions.shape[1] if actions.ndim > 1 else 1
        buffer = cls(
            state_dim=states.shape[1],
            action_dim=action_dim,
            capacity=length,
            discrete_actions=(not actions.dtype.is_floating_point),
            device=device,
        )
        buffer.load_state_dict(state_dict)
        return buffer

    @classmethod
    def from_h5py(cls, path, device=device):
        with h5py.File(path, "r") as f:
            data = {name: torchify(np.array(f[name]), device=device) for name in f.keys()}
        n_steps = len(data["rewards"])
        if "next_states" not in data:
            all_states = data["states"]
            assert len(all_states) == n_steps + 1
            data["states"] = all_states[:-1]
            data["next_states"] = all_states[1:]

        states, actions = data["states"], data["actions"]
        action_dim = actions.shape[1] if actions.ndim > 1 else 1
        buffer = cls(
            state_dim=states.shape[1],
            action_dim=action_dim,
            capacity=n_steps,
            discrete_actions=(not actions.dtype.is_floating_point),
            device=device,
        )
        buffer.extend(**{name: data[name] for name in cls.COMPONENT_NAMES})
        return buffer

    def __len__(self):
        return min(int(self._pointer.item()), self.capacity)

    def _create_buffer(self, name, dtype, shape):
        buffer = torch.empty(self.capacity, *shape, dtype=dtype, device=self.device)
        self.register_buffer(f"_{name}", buffer)
        self._bufs[name] = buffer

    def _get1(self, name):
        buf = self._bufs[name]
        if self._pointer <= self.capacity:
            return buf[: self._pointer]
        idx = self._pointer % self.capacity
        return torch.cat([buf[idx:], buf[:idx]])

    def get(self, *names, device=None, as_dict=False):
        if len(names) == 0:
            names = self.COMPONENT_NAMES
        bufs = [self._get1(name) for name in names]
        if device is not None:
            bufs = [buf.to(device) for buf in bufs]
        if as_dict:
            return dict(zip(names, bufs))
        return bufs if len(bufs) > 1 else bufs[0]

    def append(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        idx = self._pointer % self.capacity
        for name in self.COMPONENT_NAMES:
            self._bufs[name][idx] = kwargs[name].to(self.device)
        self._pointer += 1

    def extend(self, **kwargs):
        assert set(kwargs.keys()) == set(self.COMPONENT_NAMES)
        batch_size = len(list(kwargs.values())[0])
        assert batch_size <= self.capacity
        idx = self._pointer % self.capacity
        end = idx + batch_size
        if end <= self.capacity:
            for name in self.COMPONENT_NAMES:
                self._bufs[name][idx:end] = kwargs[name].to(self.device)
        else:
            fit = self.capacity - idx
            overflow = end - self.capacity
            for name in self.COMPONENT_NAMES:
                buf, arg = self._bufs[name], kwargs[name].to(self.device)
                buf[-fit:] = arg[:fit]
                buf[:overflow] = arg[-overflow:]
        self._pointer += batch_size

    def sample(self, batch_size, replace=True, device=device, include_indices=False):
        if replace:
            indices = torch.randint(len(self), [batch_size], device=self.device)
        else:
            indices = torch.randperm(len(self), device=self.device)[:batch_size]
        bufs = [self._bufs[name][indices] for name in self.COMPONENT_NAMES]
        if device is not None:
            bufs = [buf.to(device) for buf in bufs]
            out_indices = indices.to(device)
        else:
            out_indices = indices
        return (bufs, out_indices) if include_indices else bufs

    def trimmed_copy(self):
        new_buffer = self.__class__(
            self.state_dim,
            self.action_dim,
            len(self),
            discrete_actions=self.discrete_actions,
            device=self.device,
        )
        new_buffer.extend(**self.get(as_dict=True, device=None))
        return new_buffer

    def save_h5py(self, path, remove_duplicate_states=True):
        data = self.get(as_dict=True, device="cpu")
        if remove_duplicate_states:
            next_states = data.pop("next_states")
            data["states"] = torch.cat((data["states"], next_states[-1].unsqueeze(0)))
        with h5py.File(path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value.numpy())


def concat_sample_buffers(buffers):
    state_dim, action_dim = buffers[0].state_dim, buffers[0].action_dim
    discrete_actions = buffers[0].discrete_actions
    total_capacity = sum(len(buffer) for buffer in buffers)
    combined_buffer = SampleBuffer(
        state_dim,
        action_dim,
        total_capacity,
        discrete_actions=discrete_actions,
    )
    for buffer in buffers:
        combined_buffer.extend(**buffer.get(as_dict=True, device=None))
    return combined_buffer
