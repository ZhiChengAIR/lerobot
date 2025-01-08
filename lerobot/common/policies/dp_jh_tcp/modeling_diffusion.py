"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from lerobot.common.policies.dp_jh_tcp.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

# action中角度所在序号
ANGLE_IDX = [3, 4, 5, 10, 11, 12]
def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def normalize_angle_in_actions(actions):
    actions = einops.rearrange(actions,'n b d -> d b n')
    for idx in range(actions.shape[0]):
        if idx in ANGLE_IDX:
            actions[idx] = normalize_angle(actions[idx])
    actions = einops.rearrange(actions,'d b n -> n b d')
    return actions

def update_ensembled_actions(ensembled_actions,actions,alpha):


    ensembled_actions = einops.rearrange(ensembled_actions,'n b d -> d b n')
    actions = einops.rearrange(actions,'n b d -> d b n')

    for idx in range(ensembled_actions.shape[0]):
            if idx in ANGLE_IDX:
                cache = ensembled_actions[idx] + normalize_angle(normalize_angle(actions[idx]) - ensembled_actions[idx])
                ensembled_actions[idx] = normalize_angle(alpha * ensembled_actions[idx] + (1 - alpha) * cache)
            else:
                ensembled_actions[idx] = alpha * ensembled_actions[idx] + (1 - alpha) * actions[idx]
    
    ensembled_actions = einops.rearrange(ensembled_actions,'d b n -> n b d')

    return ensembled_actions

#------------------------------------------------------------#
# changed by jh
import sys
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
class FPNResNetEncoder(torch.nn.Module):
    def __init__(self, backbone_name='resnet34', pretrained=False):
        """
        Args:
            backbone_name (str): Backbone name such as 'resnet18', 'resnet34', 'resnet50'.
            pretrained (bool): Whether to use pretrained weights.
        """
        super(FPNResNetEncoder, self).__init__()
        # Initialize FPN-based ResNet backbone
        self.model = resnet_fpn_backbone(backbone_name, pretrained=pretrained)
        
        # Use out_channels directly since it's an integer, not a dictionary
        # Remove the incorrect feature_dim assignment
        # self.feature_dim = self.model.out_channels

        # Dynamically determine feature_dim by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # Assuming input size 224x224
            features = self.model(dummy_input)
            embeddings = []
            
            for level, feature in features.items():
                embedding = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1)).reshape(feature.size(0), -1)
                embeddings.append(embedding)
            
            # Concatenate embeddings from all FPN levels
            concatenated_embeddings = torch.cat(embeddings, dim=1)
            self.output_dim = concatenated_embeddings.size(1)
            self.feature_dim = self.output_dim  # Set feature_dim to match output_dim

        print(f"FPNResNetEncoder '{backbone_name}' Output Dimension: {self.output_dim}")

    def forward(self, x):
        # Extract feature maps at different levels of the FPN
        features = self.model(x)
        embeddings = []
        
        # Pool and concatenate features from different FPN levels
        for level, feature in features.items():
            embedding = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1)).reshape(feature.size(0), -1)
            embeddings.append(embedding)
        
        # Concatenate embeddings from all FPN levels
        concatenated_embeddings = torch.cat(embeddings, dim=1)
        return concatenated_embeddings
#------------------------------------------------------------#

#------------------------------------------------------------#
# modified by yzh
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
import hydra
#from yz_unit_test.visual_results import (plot_trajectories,
#                                         plot_avg_mse_with_betas)
from .visual_results import (plot_trajectories,
                                         plot_avg_mse_with_betas)
#------------------------------------------------------------#

class DiffusionPolicy(nn.Module, PyTorchModelHubMixin):
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = DiffusionConfig()
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.images")]

        # #added by yzh
        # self.epoch = 0  # Initialize epoch to zero

        # #added by yzh
        # self.out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir

        self.reset()

    #------------------------------------------------------------#
    # changed by jh
    # replace 'observation.state' with 'action_tcp'
    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.images": deque(maxlen=self.config.n_obs_steps),
            # "observation.state": deque(maxlen=self.config.n_obs_steps),
            "observation.tcppose": deque(maxlen=self.config.n_obs_steps),
            "action_tcp": deque(maxlen=self.config.n_action_steps),
            # "action": deque(maxlen=self.config.n_action_steps),
        }
    #------------------------------------------------------------#

    #------------------------------------------------------------#
    # added by yzh
    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Predict actions given observations for evaluation purposes.

        Args:
            batch (dict[str, Tensor]): A dictionary containing observation data.

        Returns:
            Tensor: Predicted actions.
        """
        # Normalize inputs
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # Prepare the batch for the diffusion model
        #batch = {k: v.unsqueeze(0) if len(v.shape) == 3 else v for k, v in batch.items()}  # Ensure batch dimension

        # Generate actions using the diffusion model
        actions = self.diffusion.generate_actions2(batch)
        
        #debug:
        #print('Generate actions using the diffusion model:',actions)

        # Unnormalize the generated actions
        actions = self.unnormalize_outputs({"action_tcp": actions})["action_tcp"]
        #actions = self.unnormalize_outputs({"action": actions})["action"]

        return actions
    #------------------------------------------------------------#

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... |n-o+1+h|
            |observation is used | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps < horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action_tcp"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action_tcp": actions})["action_tcp"]

            self._queues["action_tcp"].extend(actions.transpose(0, 1))

        action = self._queues["action_tcp"].popleft()
        return action

    @torch.no_grad
    def select_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... |n-o+1+h|
            |observation is used | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps < horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)
        n_obs_steps = self.config.n_obs_steps
        horizon = self.config.horizon
        n_action_steps = self.config.n_action_steps
        updata_len = horizon - (n_obs_steps+2) - n_action_steps

        if False:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions1(batch)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action_tcp": actions})["action_tcp"]
            actions = actions.transpose(0, 1)

            self._queues["action_tcp"].clear()
            self._queues["action_tcp"].extend(actions)

            if self._ensembled_actions is None:
                self._ensembled_actions = actions.clone()
            else:
                alpha = 0.9
                self._ensembled_actions = update_ensembled_actions(self._ensembled_actions,actions[:-1],alpha)
                # The last action, which has no prior moving average, needs to get concatenated onto the end.
                self._ensembled_actions = torch.cat([self._ensembled_actions, actions[-1:]], dim=0)
            self._queues["action_tcp"].popleft()
            action, self._ensembled_actions = self._ensembled_actions[0], self._ensembled_actions[1:]
            return action, self._ensembled_actions
        
        else:
            if len(self._queues["action_tcp"]) < 8:
                # stack n latest observations from the queue
                batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
                actions = self.diffusion.generate_actions1(batch)

                # TODO(rcadene): make above methods return output dictionary?
                actions = self.unnormalize_outputs({"action_tcp": actions})["action_tcp"]
                actions = normalize_angle_in_actions(actions)

                self._queues["action_tcp"].clear()
                self._queues["action_tcp"].extend(actions.transpose(0, 1))

        action = self._queues["action_tcp"].popleft()
        return action,self._queues["action_tcp"]

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        return {"loss": loss}


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        if self.config.vision_backbone == 'resnet18':
            print('*'*20,flush=True)
            print('vision_backbone: resnet18',flush=True)
            self.rgb_encoder = DiffusionRgbEncoder(config)
        elif self.config.vision_backbone == 'fpn_resnet34':
            print('*'*20,flush=True)
            print('vision_backbone: fpn_resnet34',flush=True)
            self.rgb_encoder = FPNResNetEncoder('resnet34', pretrained=config.pretrained_backbone_weights is not None)
        else:
            raise ValueError(f"Unsupported vision backbone {config.vision_backbone}")
        
        #added by yzh
        #self.out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir

        num_images = len([k for k in config.input_shapes if k.startswith("observation.images")])

        # self.unet = DiffusionConditionalUnet1d(
        #     config,
        #     global_cond_dim=(
        #         config.input_shapes["observation.state"][0] + self.rgb_encoder.feature_dim * num_images
        #     )
        #     * config.n_obs_steps,
        # )
        
        #------------------------------------------------------------#
        # changed by jh
        # global_cond_dim = (config.input_shapes["observation.state"][0] + self.rgb_encoder.feature_dim * num_images) * config.n_obs_steps
        print('config.input_shapes:',config.input_shapes)
        global_cond_dim = (config.input_shapes["observation.tcppose"][0] + self.rgb_encoder.feature_dim * num_images) * config.n_obs_steps

        if config.model_type == 'unet':
            self.model = DiffusionConditionalUnet1d(
                config,
                global_cond_dim=global_cond_dim,
            )
            print('*'*20,flush=True)
            print('action prediction model_type: unet',flush=True)
        elif config.model_type == 'transformer':
            self.model = DiffusionTransformer(
                config,
                global_cond_dim=global_cond_dim,
            )
            print('*'*20,flush=True)
            print('action prediction model_type: transformer',flush=True)
        else:
            raise ValueError(f"Unsupported model type {config.model_type}")

        # def count_parameters(model):
        #     num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     return num
        def count_parameters(model):
            num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return num / 1e6

        # Inside your DiffusionModel class after initialization
        print(f"RGB Encoder parameters: {count_parameters(self.rgb_encoder):.1f}M", flush=True)
        print(f"Action Prediction Network parameters: {count_parameters(self.model):.1f}M", flush=True)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.output_shapes["action_tcp"][0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.model(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        # batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        batch_size, n_obs_steps = batch["observation.tcppose"].shape[:2]
        # Extract image feature (first combine batch, sequence, and camera index dims).
        img_features = self.rgb_encoder(
            einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
        )
        # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the feature
        # dim (effectively concatenating the camera features).
        img_features = einops.rearrange(
            img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        # return torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)
        return torch.cat([batch["observation.tcppose"], img_features], dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
        }
        """
        # batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        batch_size, n_obs_steps = batch["observation.tcppose"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        #debug:
        print('generate_actions():',actions)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def generate_actions1(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
        }
        """
        # batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        batch_size, n_obs_steps = batch["observation.tcppose"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        #debug:
        print('generate_actions1():',actions)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps + 2
        # end = start + self.config.n_action_steps
        actions = actions[:, start:]

        return actions

    #------------------------------------------------------------#
    # added by yzh
    def generate_actions2(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
        }
        """
        #batch_size, n_obs_steps = batch["observation.state"].shape[:2] 
        batch_size, n_obs_steps = batch["observation.tcppose"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)
        
        #debug:
        #print('generate_actions2():',actions)

        return actions
    #------------------------------------------------------------#

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        # assert set(batch).issuperset({"observation.state", "observation.images", "action", "action_is_pad"})
        assert set(batch).issuperset({"observation.tcppose", "observation.images", "action_tcp", "action_tcp_is_pad"})
        # n_obs_steps = batch["observation.state"].shape[1]
        n_obs_steps = batch["observation.tcppose"].shape[1]
        horizon = batch["action_tcp"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action_tcp"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action_tcp"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none") 

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_tcp_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_tcp_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        #out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir
        # plot_trajectories(trajectory, #gt_action,
        #                                   pred, #pred_action,
        #                                   #self.epoch,
        #                                   self.out_dir)
        
        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.input_shapes` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.input_shapes`.
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x

#------------------------------------------------------------#
# changed by jh
# class DiffusionRgbEncoder(nn.Module):
#     """Encoder an RGB image into a 1D feature vector, handling multiple camera views."""

#     def __init__(self, config: DiffusionConfig):
#         super().__init__()
#         self.config = config

#         # Set up optional preprocessing.
#         if config.crop_shape is not None:
#             self.do_crop = True
#             self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
#             self.maybe_random_crop = (
#                 torchvision.transforms.RandomCrop(config.crop_shape)
#                 if config.crop_is_random
#                 else self.center_crop
#             )
#         else:
#             self.do_crop = False

#         # Collect all image keys
#         image_keys = [k for k in config.input_shapes if k.startswith("observation.images")]
#         if not image_keys:
#             raise ValueError("No observation.images keys found in input_shapes.")

#         # Use the first image key to create a dummy input
#         first_image_key = image_keys[0]
#         dummy_input = torch.zeros(
#             1,
#             config.input_shapes[first_image_key][0],
#             *config.input_shapes[first_image_key][1:]
#         )

#         # Set up backbone
#         if config.vision_backbone.startswith("fpn_resnet"):
#             # FPN-based ResNet backbone
#             resnet_backbone_name = config.vision_backbone.replace("fpn_", "")
#             self.backbone = resnet_fpn_backbone(resnet_backbone_name, pretrained=config.pretrained_backbone_weights)
#             self.feature_dim = sum(self.backbone.out_channels)  # Summing FPN output levels

#             # Since FPN outputs multiple feature maps, use adaptive pooling and concatenate
#             self.pool = nn.Sequential(
#                 nn.AdaptiveAvgPool2d((1, 1)),
#                 nn.Flatten(),
#             )
#             self.out = nn.Linear(self.feature_dim, self.feature_dim)
#             self.relu = nn.ReLU()
#         else:
#             # Standard ResNet backbone
#             backbone_model = getattr(torchvision.models, config.vision_backbone)(
#                 weights=config.pretrained_backbone_weights
#             )
#             self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))  # Remove FC layer
#             self.feature_dim = backbone_model.fc.in_features  # Use last layer's input features for dim

#             # Replace BatchNorm with GroupNorm if required.
#             if config.use_group_norm:
#                 if config.pretrained_backbone_weights:
#                     raise ValueError("Can't replace BatchNorm in pretrained model without ruining the weights!")
#                 self.backbone = _replace_submodules(
#                     root_module=self.backbone,
#                     predicate=lambda x: isinstance(x, nn.BatchNorm2d),
#                     func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
#                 )

#             # Determine feature map shape using a dummy input
#             dummy_feature_map = self.backbone(dummy_input)
#             feature_map_shape = tuple(dummy_feature_map.shape[1:])  # [C, H, W]

#             self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
#             self.feature_dim = config.spatial_softmax_num_keypoints * 2
#             self.out = nn.Linear(self.feature_dim, self.feature_dim)
#             self.relu = nn.ReLU()

#         # Final layer
#         self.final_layer = nn.Sequential(
#             self.relu,
#             self.out
#         )

#     def forward(self, x: dict[str, Tensor]) -> Tensor:
#         """Forward pass to encode the input images."""
#         # Preprocess the images.
#         if self.do_crop:
#             x = {k: (self.maybe_random_crop(v) if self.training else self.center_crop(v)) for k, v in x.items()}

#         # Extract backbone features from all camera images
#         image_keys = [k for k in self.config.input_shapes if k.startswith("observation.images")]
#         features = []
#         for key in image_keys:
#             img = x[key]  # Assuming x is a dict with keys as image keys
#             feat = self.backbone(img)
#             features.append(feat)

#         if self.config.vision_backbone.startswith("fpn_resnet"):
#             # Apply adaptive pooling and concatenate embeddings
#             embeddings = []
#             for feat in features:
#                 emb = self.pool(feat)
#                 embeddings.append(emb)
#             x = torch.cat(embeddings, dim=1)  # [B, feature_dim * num_cameras]
#         else:
#             # Standard ResNet: Apply SpatialSoftmax
#             # Assuming a single image per sample
#             feat = features[0]  # [B, C, H, W]
#             x = self.pool(feat)  # [B, K, 2]
#             x = torch.flatten(x, start_dim=1)  # [B, K*2]

#         # Final linear layer with non-linearity.
#         x = self.final_layer(x)  # [B, feature_dim]
#         return x
#------------------------------------------------------------#

def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.output_shapes["action_tcp"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action_tcp"][0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

#------------------------------------------------------------#
# changed by jh
# Class DiffusionTransformer

from typing import Union, Optional, Tuple
import torch
import torch.nn as nn

class DiffusionTransformer(nn.Module):
    """
    Transformer-based model for action prediction in the diffusion policy.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Initialize model parameters from config
        self.horizon = config.horizon
        self.n_obs_steps = config.n_obs_steps
        self.input_dim = config.input_shapes["observation.tcppose"][0]
        self.output_dim = config.output_shapes["action_tcp"][0]
        self.cond_dim = global_cond_dim

        # Transformer parameters from config
        n_layer = config.transformer_n_layer
        n_head = config.transformer_n_head
        n_emb = config.transformer_n_emb
        p_drop_emb = config.transformer_p_drop_emb
        p_drop_attn = config.transformer_p_drop_attn
        causal_attn = config.transformer_causal_attn
        time_as_cond = config.transformer_time_as_cond
        obs_as_cond = config.transformer_obs_as_cond
        n_cond_layers = config.transformer_n_cond_layers

        # Initialize the TransformerForDiffusion model
        self.transformer = TransformerForDiffusion(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            cond_dim=self.cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the transformer.
            timestep: (B,) tensor of timesteps.
            global_cond: (B, global_cond_dim)
        Returns:
            (B, T, output_dim) diffusion model prediction.
        """
        # The TransformerForDiffusion expects the conditioning input as 'cond'
        # We will pass the 'global_cond' as 'cond'

        return self.transformer(sample=x, timestep=timestep, cond=global_cond)

# Include the TransformerForDiffusion class from your provided script
# Ensure to import any necessary modules and dependencies

class TransformerForDiffusion(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 horizon: int,
                 n_obs_steps: int,
                 cond_dim: int,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_emb: int = 768,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool = False,
                 time_as_cond: bool = True,
                 obs_as_cond: bool = False,
                 n_cond_layers: int = 0
                 ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        # Compute number of tokens for main trunk and condition encoder
        T = horizon
        T_cond = 1  # For timestep

        if not time_as_cond:
            T += 1
            T_cond -= 1

        obs_as_cond = cond_dim > 0
        if time_as_cond and obs_as_cond:
            T_cond += n_obs_steps  # Total: 3
        elif obs_as_cond:
            T_cond += n_obs_steps  # Adjust if time_as_cond is False

        # Input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Cond encoder
        self.time_emb = DiffusionSinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim // n_obs_steps, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # Decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True  # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # Encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # Attention mask
        if causal_attn:
            # Causal mask to ensure that attention is only applied to the left
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')) \
                         .masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s - 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')) \
                         .masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # Decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        #self.head = FinalLayer(n_emb, output_dim) #modified by yzh

        # Constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        DiffusionSinusoidalPosEmb,
                        ############added by yz
                        RmsNorm, 
                        Mlp, 
                        FinalLayer, 
                        nn.GELU, 
                        nn.Identity, #not necessary for jh's code though, dont know why
                        ##############
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight',
                'q_proj_weight',
                'k_proj_weight',
                'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

                bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
                for name in bias_names:
                    bias = getattr(module, name)
                    if bias is not None:
                        torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                cond: Optional[torch.Tensor] = None, **kwargs):
        """
        sample: (B, T, input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, T_cond, cond_dim)
        output: (B, T, output_dim)
        """
        # 1. Time Embedding
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timesteps = timestep.unsqueeze(0).to(sample.device)
        else:
            timesteps = timestep.to(sample.device)
        timesteps_embed = self.time_emb(timesteps).unsqueeze(1)  # Shape: (B,1,n_emb)

        # 2. Input Embedding
        input_emb = self.input_emb(sample)  # Shape: (B,T,n_emb)

        # 3. Conditioning
        if self.obs_as_cond and cond is not None:
            # batch_size, n_obs_steps = sample.shape[:2]
            batch_size = sample.shape[0]
            n_obs_steps = self.n_obs_steps
            cond = cond.view(batch_size, n_obs_steps, -1)  # Shape: (B,2,cond_dim_per_step)
            cond_obs_emb = self.cond_obs_emb(cond)  # Shape: (B,2,n_emb)
            cond_embeddings = torch.cat([timesteps_embed, cond_obs_emb], dim=1)  # Shape: (B,3,n_emb)
        else:
            cond_embeddings = timesteps_embed  # Shape: (B,1,n_emb)

        # 4. Position Embedding for Conditioning
        if self.cond_pos_emb is not None:
            position_embeddings = self.cond_pos_emb[:, :cond_embeddings.shape[1], :]
            x = self.drop(cond_embeddings + position_embeddings)  # Shape: (B,3,n_emb)
            x = self.encoder(x)  # Shape depends on encoder
            memory = x  # Shape: (B,3,n_emb)
        else:
            memory = None

        # 5. Input Sequence Position Embedding
        position_embeddings = self.pos_emb[:, :sample.shape[1], :]
        x = self.drop(input_emb + position_embeddings)  # Shape: (B,16,n_emb)

        # 6. Transformer Decoder
        if self.encoder_only:
            # BERT-style
            x = self.encoder(x, mask=self.mask)
        else:
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )

        # 7. Output Head
        x = self.ln_f(x)
        x = self.head(x)  # Shape: (B,16,output_dim)

        return x

#------------------------------------------------------------#
# modified by yzh
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, n_emb, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(n_emb, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=n_emb,
            hidden_features=n_emb,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x
#------------------------------------------------------------#