#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.meta_episode_dones = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, actor_critic, device="cpu"):
        # store inputs
        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.meta_episode_dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        
        # RNN hidden states for symmetry augmentation
        self.augmented_hidden_states = None
        self.augmented_last_critics_obs = None
        
        # We need the actor critic to perform forward pass to obtain the augmented hidden states
        self.actor_critic = actor_critic

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        
        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        
        # For RNN networks
        self._save_hidden_states(transition.hidden_states)
        self.meta_episode_dones[self.step].copy_(transition.meta_episode_dones.view(-1, 1))
        
        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()
    
    def get_augmented_hidden_states(self, observations, critic_observations, dones):
        seq_len, augmented_batch_size = observations.shape[:2]
        dones = dones.squeeze(-1).bool()

        if self.augmented_hidden_states is not None:
            hidden_actor, hidden_critic = self.augmented_hidden_states
        else:
            hidden_actor, hidden_critic = self.actor_critic.init_hidden(augmented_batch_size, device=self.device)

        # Forward pass through the RNN to obtain hidden states for augmented samples
        augmented_hidden_actor = []
        augmented_hidden_critic = []
        # Make sure that the forward pass does not track gradients
        with torch.inference_mode():
            for t in range(seq_len):
                augmented_hidden_actor.append(hidden_actor)
                augmented_hidden_critic.append(hidden_critic)
                self.actor_critic.act_inference(observations[t], hidden_states=hidden_actor)
                self.actor_critic.evaluate(critic_observations[t], hidden_states=hidden_critic)
                hidden_actor, hidden_critic = self.actor_critic.get_hidden_out()
                for j in range(len(hidden_actor)):
                    hidden_actor[j][:, dones[t]] = 0.0
                    hidden_critic[j][:, dones[t]] = 0.0

        # Store the final hidden states for use in the next epoch
        self.augmented_hidden_states = (hidden_actor, hidden_critic)

        # TODO: adjust for gru
        hid_states_actor_batch = [torch.stack([h[0] for h in augmented_hidden_actor], dim=0),
                                torch.stack([h[1] for h in augmented_hidden_actor], dim=0)]
        hid_states_critic_batch = [torch.stack([h[0] for h in augmented_hidden_critic], dim=0),
                                torch.stack([h[1] for h in augmented_hidden_critic], dim=0)]
        hidden_states_actor = [torch.cat((self.saved_hidden_states_a[i], hid_states_actor_batch[i][..., self.num_envs:, :]), dim=2) for i in range(len(self.saved_hidden_states_a))]
        hidden_states_critic = [torch.cat((self.saved_hidden_states_c[i], hid_states_critic_batch[i][..., self.num_envs:, :]), dim=2) for i in range(len(self.saved_hidden_states_c))]

        return hidden_states_actor, hidden_states_critic

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                
                # Yield the mini-batch
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8, symmetry_cfg=None):
        num_aug = 1
        if symmetry_cfg is not None and symmetry_cfg["use_data_augmentation"]:
            # Get the augmented observations and actions
            data_augmentation_func = symmetry_cfg["data_augmentation_func"]
            observations, actions = data_augmentation_func(obs=self.observations, actions=self.actions, env=symmetry_cfg["env"])
                        
            if self.privileged_observations is not None:
                critic_observations, _ = data_augmentation_func(
                    obs=self.privileged_observations, actions=None, env=symmetry_cfg["env"], is_critic=True
                )
            else:
                critic_observations = observations
                        
            # compute number of augmentations per sample
            num_aug = int(observations.shape[1] / self.num_envs)
            dones = self.dones.repeat(1, num_aug)
            meta_episode_dones = self.meta_episode_dones.repeat(1, num_aug, 1)
            
            # Get the augmented hidden states
            hidden_states_actor, hidden_states_critic = self.get_augmented_hidden_states(observations, critic_observations, meta_episode_dones)
        else:
            observations = self.observations
            critic_observations = self.privileged_observations if self.privileged_observations is not None else self.observations
            actions = self.actions
            dones = self.dones.squeeze(-1)
            
            hidden_states_actor = self.saved_hidden_states_a
            hidden_states_critic = self.saved_hidden_states_c
            
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(observations, dones)
        padded_critic_obs_trajectories, _ = split_and_pad_trajectories(critic_observations, dones)
        
        original_batch_size = padded_obs_trajectories.shape[1] // num_aug

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                # Concatenate the desired trajectory indices
                traj_indices = torch.arange(first_traj, last_traj)
                for k in range(1, num_aug):
                    traj_batch_start = k * original_batch_size
                    traj_indices = torch.cat((traj_indices, torch.arange(traj_batch_start + first_traj, traj_batch_start + last_traj)))

                masks_batch = trajectory_masks[:, traj_indices]
                obs_batch = padded_obs_trajectories[:, traj_indices]
                critic_obs_batch = padded_critic_obs_trajectories[:, traj_indices]
                
                # Concatenate the desired batch indices
                indices = torch.arange(start, stop)
                for k in range(1, num_aug):
                    batch_start = k * self.num_envs
                    indices = torch.cat((indices, torch.arange(batch_start + start, batch_start + stop)))

                actions_batch = actions[:, indices]
                
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                
                # Make sure the data required for updates are repeated for the augmented samples
                returns_batch = self.returns[:, start:stop].repeat(1, num_aug, 1)
                advantages_batch = self.advantages[:, start:stop].repeat(1, num_aug, 1)
                values_batch = self.values[:, start:stop].repeat(1, num_aug, 1)
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop].repeat(1, num_aug, 1)

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][traj_indices]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in hidden_states_actor
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][traj_indices]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in hidden_states_critic
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj
