"""A compact, language-conditioned diffusion policy for LIBERO action blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from diffusers import DDIMScheduler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    """Fixed dimensions for the fast-reproduction diffusion policy."""

    visual_dim: int = 512
    state_dim: int = 8
    obs_horizon: int = 2
    action_dim: int = 7
    action_horizon: int = 8
    instruction_embedding_dim: int = 32
    language_dim: int = 128
    condition_dim: int = 256
    timestep_dim: int = 64
    hidden_dim: int = 512
    max_instruction_bytes: int = 256
    num_train_timesteps: int = 50


class ByteInstructionEncoder(nn.Module):
    """Encode padded byte-token tensors using their unpadded lengths."""

    padding_index = 256

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        max_bytes: int = 256,
    ) -> None:
        super().__init__()
        self.max_bytes = max_bytes
        self.embedding = nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_dim, padding_idx=self.padding_index
        )
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor with shape [B, L]")
        if lengths.shape != (tokens.shape[0],) or lengths.dtype != torch.long:
            raise ValueError("lengths must be a LongTensor with shape [B]")
        if torch.any(lengths <= 0) or torch.any(lengths > tokens.shape[1]):
            raise ValueError("lengths must be in [1, L]")
        if tokens.device != self.embedding.weight.device:
            raise ValueError("tokens and encoder must be on the same device")
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class _ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = F.silu(self.first(inputs))
        return inputs + self.second(F.silu(residual))


class CompactDiffusionPolicy(nn.Module):
    """Predict normalized LIBERO action blocks conditioned on two observations."""

    def __init__(self, config: DiffusionPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionPolicyConfig()
        self.instruction_encoder = ByteInstructionEncoder(
            embedding_dim=self.config.instruction_embedding_dim,
            hidden_dim=self.config.language_dim,
            max_bytes=self.config.max_instruction_bytes,
        )
        condition_input_dim = (
            self.config.obs_horizon * self.config.visual_dim
            + self.config.obs_horizon * self.config.state_dim
            + self.config.language_dim
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_input_dim, self.config.condition_dim),
            nn.SiLU(),
            nn.Linear(self.config.condition_dim, self.config.condition_dim),
            nn.SiLU(),
        )
        action_block_dim = self.config.action_horizon * self.config.action_dim
        self.input_projection = nn.Linear(
            action_block_dim + self.config.condition_dim + self.config.timestep_dim,
            self.config.hidden_dim,
        )
        self.residual_blocks = nn.ModuleList(
            [_ResidualMLPBlock(self.config.hidden_dim) for _ in range(4)]
        )
        self.output_projection = nn.Linear(self.config.hidden_dim, action_block_dim)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

    def _validate_observations(self, visual: Tensor, state: Tensor) -> None:
        expected_visual = (
            visual.ndim == 3
            and visual.shape[1:] == (self.config.obs_horizon, self.config.visual_dim)
        )
        expected_state = (
            state.ndim == 3
            and state.shape[1:] == (self.config.obs_horizon, self.config.state_dim)
        )
        if not expected_visual or not expected_state or visual.shape[0] != state.shape[0]:
            raise ValueError("visual/state must be matching [B, obs_horizon, feature] tensors")

    def encode_instruction(self, instructions: Sequence[str]) -> Tensor:
        """Tokenize UTF-8 strings and return one 128-d embedding per instruction."""

        if not instructions:
            raise ValueError("instructions must not be empty")
        encoded_rows: list[bytes] = []
        for instruction in instructions:
            if not isinstance(instruction, str):
                raise TypeError("each instruction must be a string")
            encoded_rows.append(
                instruction.encode("utf-8")[: self.config.max_instruction_bytes]
            )
        lengths = torch.tensor(
            [max(1, len(row)) for row in encoded_rows],
            dtype=torch.long,
            device=self.instruction_encoder.embedding.weight.device,
        )
        width = int(lengths.max().item())
        tokens = torch.full(
            (len(encoded_rows), width),
            self.instruction_encoder.padding_index,
            dtype=torch.long,
            device=lengths.device,
        )
        for row_index, encoded in enumerate(encoded_rows):
            if encoded:
                tokens[row_index, : len(encoded)] = torch.tensor(
                    list(encoded), dtype=torch.long, device=tokens.device
                )
        return self.instruction_encoder(tokens, lengths)

    def _condition(self, condition: Mapping[str, object]) -> Tensor:
        try:
            visual = condition["visual"]
            state = condition["state"]
            instructions = condition["instructions"]
        except KeyError as error:
            raise ValueError(f"condition is missing {error.args[0]!r}") from error
        if not isinstance(visual, Tensor) or not isinstance(state, Tensor):
            raise TypeError("condition visual/state values must be tensors")
        if not isinstance(instructions, Sequence) or isinstance(instructions, str):
            raise TypeError("condition instructions must be a sequence of strings")
        self._validate_observations(visual, state)
        if len(instructions) != visual.shape[0]:
            raise ValueError("instructions must have one entry per batch row")
        language = self.encode_instruction(instructions).to(dtype=visual.dtype)
        features = torch.cat((visual.flatten(1), state.flatten(1), language), dim=-1)
        return self.condition_projection(features)

    def _timestep_embedding(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor:
        half_dim = self.config.timestep_dim // 2
        scale = math.log(10_000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=dtype) * -scale
        )
        angles = timesteps.to(dtype=dtype).unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def _predict_noise(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        condition: Mapping[str, object],
    ) -> Tensor:
        expected_shape = (
            noisy_actions.ndim == 3
            and noisy_actions.shape[1:]
            == (self.config.action_horizon, self.config.action_dim)
        )
        if not expected_shape:
            raise ValueError("noisy_actions must have shape [B, action_horizon, action_dim]")
        if timesteps.shape != (noisy_actions.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        condition_features = self._condition(condition)
        time_embedding = self._timestep_embedding(timesteps, noisy_actions.dtype)
        features = torch.cat(
            (
                noisy_actions.flatten(1),
                condition_features.to(noisy_actions.dtype),
                time_embedding,
            ),
            dim=-1,
        )
        hidden = F.silu(self.input_projection(features))
        for block in self.residual_blocks:
            hidden = block(hidden)
        return self.output_projection(F.silu(hidden)).view_as(noisy_actions)

    def training_loss(
        self,
        batch: Mapping[str, object],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample a timestep/noise per row and return scalar epsilon MSE."""

        try:
            condition = batch["condition"]
            actions = batch["actions"]
        except KeyError as error:
            raise ValueError(f"batch is missing {error.args[0]!r}") from error
        if not isinstance(condition, Mapping):
            raise TypeError("batch condition must be a mapping")
        if not isinstance(actions, Tensor):
            raise TypeError("batch actions must be a tensor")
        batch_size = actions.shape[0]
        timesteps = torch.randint(
            0,
            self.config.num_train_timesteps,
            (batch_size,),
            device=actions.device,
            generator=generator,
        )
        noise = torch.randn(
            actions.shape, device=actions.device, dtype=actions.dtype, generator=generator
        )
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        prediction = self._predict_noise(noisy_actions, timesteps, condition)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample_actions(
        self,
        condition: Mapping[str, object],
        num_inference_steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """DDIM sample a normalized action block, deterministically for a seeded generator."""

        try:
            visual = condition["visual"]
            state = condition["state"]
        except KeyError as error:
            raise ValueError(f"condition is missing {error.args[0]!r}") from error
        if not isinstance(visual, Tensor) or not isinstance(state, Tensor):
            raise TypeError("condition visual/state values must be tensors")
        self._validate_observations(visual, state)
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        batch_size = visual.shape[0]
        actions = torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=visual.device,
            dtype=visual.dtype,
            generator=generator,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=actions.device)
        for timestep in self.scheduler.timesteps:
            batched_timestep = torch.full(
                (batch_size,), int(timestep.item()), device=actions.device, dtype=torch.long
            )
            noise_prediction = self._predict_noise(actions, batched_timestep, condition)
            actions = self.scheduler.step(
                noise_prediction, timestep, actions, generator=generator
            ).prev_sample
        return actions.clamp(-1, 1)


DiffusionPolicy = CompactDiffusionPolicy
