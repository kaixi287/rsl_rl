#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        symmetry_cfg: dict | None = None,
        **kwargs,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            self.use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not self.use_symmetry:
                print("Symmetry configuration is provided but not used.")
            # If funciton is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Data augmentation enalbled but the function is not callable: {symmetry_cfg['data_augmentation_func']}"
                )
            # store symmetry configuration
            self.symmetry_cfg = symmetry_cfg
        else:
            self.use_symmetry = False
            self.symmetry_cfg = None
            
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.obs_history = None
        self.critic_obs_history = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.actor_critic, self.device
        )
        if self.actor_critic.with_history:
            self.obs_history = torch.zeros(
                (num_envs, self.actor_critic.history_len, *actor_obs_shape), dtype=torch.float32, device=self.device
            )
            self.critic_obs_history = torch.zeros(
                (num_envs, self.actor_critic.history_len, *critic_obs_shape), dtype=torch.float32, device=self.device
            )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def get_stacked_obs(self, new_obs, new_critic_obs):
        if self.obs_history is None or self.critic_obs_history is None:
            raise ValueError("Obs history is not initialized. Cannot stack observations.")
        # Shift left and insert the latest obs
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = new_obs
        stacked_obs = self.obs_history.reshape(new_obs.shape[0], -1)
        # Shift left and insert the latest critic obs
        self.critic_obs_history = torch.roll(self.critic_obs_history, shifts=-1, dims=1)
        self.critic_obs_history[:, -1, :] = new_critic_obs
        stacked_critic_obs = self.critic_obs_history.reshape(new_critic_obs.shape[0], -1)
        return stacked_obs, stacked_critic_obs

    def reset_obs_history(self, dones):
        if self.obs_history is None or self.critic_obs_history is None:
            return
        # Reset history where done=True
        dones = dones.view(-1)
        for env_idx in range(dones.shape[0]):
            if dones[env_idx]:
                self.obs_history[env_idx] = 0.0

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        if self.actor_critic.with_history:
            stacked_obs, stacked_critic_obs = self.get_stacked_obs(obs, critic_obs)
            self.transition.actions = self.actor_critic.act(stacked_obs).detach()
            self.transition.values = self.actor_critic.evaluate(stacked_critic_obs).detach()
        else:
            # Compute the actions and values
            self.transition.actions = self.actor_critic.act(obs).detach()
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, meta_episode_dones=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        if self.actor_critic.with_history:
            # Reset history for done environments
            self.reset_obs_history(meta_episode_dones if meta_episode_dones is not None else dones)

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        
        # Process the meta-episode dones and reset done envs
        if meta_episode_dones is not None:
            self.transition.meta_episode_dones = meta_episode_dones
            # Reset the actor-critic hidden states according to meta-episode dones
            self.actor_critic.reset(meta_episode_dones)
        else:
            self.transition.meta_episode_dones = dones
            # Reset the actor-critic hidden states according to nominal eipsode dones
            self.actor_critic.reset(dones)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        # Note that we modify here to not update the hidden states
        last_critic_hidden = None
        if self.actor_critic.is_recurrent:
            last_critic_hidden = self.actor_critic.get_hidden_states()[1]
        last_values = self.actor_critic.evaluate(last_critic_obs, hidden_states=last_critic_hidden).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        #  -- Symmetry loss
        if self.use_symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        if self.actor_critic.is_recurrent or self.actor_critic.with_history:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs, self.symmetry_cfg)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            
            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = old_mu_batch.shape[-2]
            
            # Perform symmetric augmentation
            if self.use_symmetry and self.symmetry_cfg["use_data_augmentation"] and not self.actor_critic.is_recurrent:
                # augmentation using symmetry
                # returned shape: [batch_size * num_aug, ...]
                data_augmentation_func = self.symmetry_cfg["data_augmentation_func"]
                obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry_cfg["env"])
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry_cfg["env"], is_critic=True
                )
                _, actions_batch = data_augmentation_func(
                    obs=None, actions=actions_batch, env=self.symmetry_cfg["env"]
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[-2] / original_batch_size)
                # repeat the rest of the batch
                repeat_dims = [1] * (obs_batch.dim() - 1)
                repeat_dims.insert(-1, num_aug)
                # --actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(*repeat_dims)
                # --critic
                target_values_batch = target_values_batch.repeat(*repeat_dims)
                advantages_batch = advantages_batch.repeat(*repeat_dims)
                returns_batch = returns_batch.repeat(*repeat_dims)
                
            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to fo this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # --entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[..., :original_batch_size, :]
            sigma_batch = self.actor_critic.action_std[..., :original_batch_size, :]
            entropy_batch = self.actor_critic.entropy[..., :original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.use_symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                # this is if we want to do augmentation and mirror loss
                if not self.symmetry_cfg["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry_cfg["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry_cfg["env"]
                    )
                    _, actions_batch = data_augmentation_func(
                        obs=None, actions=actions_batch, env=self.symmetry_cfg["env"]
                    )
                # actions predicted by the actor
                pred_actions_batch = self.actor_critic.act_inference(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                # compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(pred_actions_batch, actions_batch)
                # add the loss to the total loss
                if self.symmetry_cfg["use_mirror_loss"]:
                    loss += self.symmetry_cfg["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            
            # -- Symmetry loss
            if self.use_symmetry and mean_surrogate_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_symmetry_loss
