from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union

import numpy as np

from dataset.metaworld.collector.config import CollectionCfg


@dataclass(frozen=True)
class ExpertGuidedEpisodeConfig:
    mode: str
    noise_strength: float
    noise_smoothing: float
    noise_target_hold_steps: int


@dataclass(frozen=True)
class PerturbRecoverEpisodeConfig:
    mode: str
    perturb_start: int
    perturb_duration: int
    perturb_strength: float
    noise_smoothing: float
    noise_target_hold_steps: int
    minimum_recovery_steps: int

    @property
    def perturb_stop(self) -> int:
        return self.perturb_start + self.perturb_duration


@dataclass(frozen=True)
class SmoothRandomEpisodeConfig:
    mode: str
    xyz_smoothing: float
    gripper_smoothing: float
    xyz_target_hold_steps: int
    gripper_target_hold_steps: int


EpisodeConfig = Union[
    ExpertGuidedEpisodeConfig,
    PerturbRecoverEpisodeConfig,
    SmoothRandomEpisodeConfig,
]


def episode_config_metadata(config: EpisodeConfig) -> dict[str, object]:
    return {f"action_{key}": value for key, value in asdict(config).items()}


def _sample_integer(
    rng: np.random.Generator,
    bounds: tuple[int, int],
) -> int:
    low, high = map(int, bounds)
    return int(rng.integers(low, high + 1))


def build_episode_configs(
    mode: str,
    collection: CollectionCfg,
    max_steps: int,
    rng: np.random.Generator,
) -> list[EpisodeConfig]:
    """Sample episode parameters while preserving every configured mode count."""
    if mode == "expert_guided":
        configs: list[EpisodeConfig] = []
        for entry in collection.expert_guided.noise_schedule:
            configs.extend(
                ExpertGuidedEpisodeConfig(
                    mode=mode,
                    noise_strength=float(entry.strength),
                    noise_smoothing=float(collection.expert_guided.noise_smoothing),
                    noise_target_hold_steps=_sample_integer(
                        rng, collection.expert_guided.noise_target_hold_steps
                    ),
                )
                for _ in range(int(entry.num_demos))
            )
        if len(configs) != collection.expert_guided.num_demos:
            raise ValueError(
                "The expert-guided noise schedule contains "
                f"{len(configs)} episodes, expected "
                f"{collection.expert_guided.num_demos}."
            )
        rng.shuffle(configs)
        return configs

    if mode == "perturb_recover":
        settings = collection.perturb_recover
        duration_low, duration_high = map(int, settings.duration_steps)
        start_low = int(round(float(settings.start_fraction[0]) * max_steps))
        start_high = int(round(float(settings.start_fraction[1]) * max_steps))
        latest_start = max(
            0,
            int(max_steps) - duration_high - int(settings.minimum_recovery_steps),
        )
        start_low = min(max(0, start_low), latest_start)
        start_high = min(max(start_low, start_high), latest_start)
        return [
            PerturbRecoverEpisodeConfig(
                mode=mode,
                perturb_start=int(rng.integers(start_low, start_high + 1)),
                perturb_duration=int(rng.integers(duration_low, duration_high + 1)),
                perturb_strength=float(
                    rng.uniform(
                        float(settings.strength_range[0]),
                        float(settings.strength_range[1]),
                    )
                ),
                noise_smoothing=float(settings.noise_smoothing),
                noise_target_hold_steps=_sample_integer(
                    rng, settings.noise_target_hold_steps
                ),
                minimum_recovery_steps=int(settings.minimum_recovery_steps),
            )
            for _ in range(settings.num_demos)
        ]

    if mode == "smooth_random":
        settings = collection.smooth_random
        return [
            SmoothRandomEpisodeConfig(
                mode=mode,
                xyz_smoothing=float(settings.xyz_smoothing),
                gripper_smoothing=float(settings.gripper_smoothing),
                xyz_target_hold_steps=_sample_integer(
                    rng, settings.xyz_target_hold_steps
                ),
                gripper_target_hold_steps=_sample_integer(
                    rng, settings.gripper_target_hold_steps
                ),
            )
            for _ in range(settings.num_demos)
        ]

    raise ValueError(f"Unknown collection mode: {mode!r}.")


class _ActionGenerator:
    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if self.action_low.shape != (4,) or self.action_high.shape != (4,):
            raise ValueError("Meta-World action bounds must each contain four values.")
        self.rng = rng

    def _clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_low, self.action_high).astype(
            np.float32, copy=False
        )

    def allow_success_termination(self, timestep: int) -> bool:
        return True


class _SmoothCartesianNoise:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        smoothing: float,
        target_hold_steps: int,
        rng: np.random.Generator,
    ) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.smoothing = float(smoothing)
        self.target_hold_steps = max(1, int(target_hold_steps))
        self.rng = rng
        self.target = self.rng.uniform(self.low, self.high).astype(np.float32)
        self.value = self.target.copy()

    def sample(self, timestep: int) -> np.ndarray:
        if timestep > 0 and timestep % self.target_hold_steps == 0:
            self.target = self.rng.uniform(self.low, self.high).astype(np.float32)
        self.value = (
            self.smoothing * self.value
            + (1.0 - self.smoothing) * self.target
        ).astype(np.float32)
        return self.value


class ExpertGuidedActionGenerator(_ActionGenerator):
    def __init__(
        self,
        config: ExpertGuidedEpisodeConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(action_low, action_high, rng)
        self.config = config
        self.noise = _SmoothCartesianNoise(
            self.action_low[:3],
            self.action_high[:3],
            config.noise_smoothing,
            config.noise_target_hold_steps,
            rng,
        )

    def get_action(self, expert_action: np.ndarray, timestep: int) -> np.ndarray:
        expert = np.asarray(expert_action, dtype=np.float32).reshape(4)
        beta = float(self.config.noise_strength)
        action = expert.copy()
        if beta > 0.0:
            action[:3] = (1.0 - beta) * expert[:3] + beta * self.noise.sample(
                timestep
            )
        return self._clip(action)


class PerturbRecoverActionGenerator(_ActionGenerator):
    def __init__(
        self,
        config: PerturbRecoverEpisodeConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(action_low, action_high, rng)
        self.config = config
        self.noise = _SmoothCartesianNoise(
            self.action_low[:3],
            self.action_high[:3],
            config.noise_smoothing,
            config.noise_target_hold_steps,
            rng,
        )

    def get_action(self, expert_action: np.ndarray, timestep: int) -> np.ndarray:
        expert = np.asarray(expert_action, dtype=np.float32).reshape(4)
        action = expert.copy()
        if self.config.perturb_start <= timestep < self.config.perturb_stop:
            beta = float(self.config.perturb_strength)
            action[:3] = (1.0 - beta) * expert[:3] + beta * self.noise.sample(
                timestep - self.config.perturb_start
            )
        return self._clip(action)

    def allow_success_termination(self, timestep: int) -> bool:
        # Do not end a rollout before its one required perturbation was applied.
        return timestep + 1 >= (
            self.config.perturb_stop + self.config.minimum_recovery_steps
        )


class SmoothRandomActionGenerator(_ActionGenerator):
    def __init__(
        self,
        config: SmoothRandomEpisodeConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(action_low, action_high, rng)
        self.config = config
        self.xyz_target = rng.uniform(
            self.action_low[:3], self.action_high[:3]
        ).astype(np.float32)
        self.gripper_target = float(
            rng.uniform(self.action_low[3], self.action_high[3])
        )
        self.xyz_action = np.zeros(3, dtype=np.float32)
        self.gripper_action = np.float32(0.0)

    def get_action(self, expert_action: np.ndarray, timestep: int) -> np.ndarray:
        del expert_action
        if timestep > 0 and timestep % self.config.xyz_target_hold_steps == 0:
            self.xyz_target = self.rng.uniform(
                self.action_low[:3], self.action_high[:3]
            ).astype(np.float32)
        if timestep > 0 and timestep % self.config.gripper_target_hold_steps == 0:
            self.gripper_target = float(
                self.rng.uniform(self.action_low[3], self.action_high[3])
            )
        self.xyz_action = (
            self.config.xyz_smoothing * self.xyz_action
            + (1.0 - self.config.xyz_smoothing) * self.xyz_target
        ).astype(np.float32)
        self.gripper_action = np.float32(
            self.config.gripper_smoothing * self.gripper_action
            + (1.0 - self.config.gripper_smoothing) * self.gripper_target
        )
        return self._clip(
            np.concatenate(
                (self.xyz_action, np.asarray([self.gripper_action], np.float32))
            )
        )


def build_action_generator(
    config: EpisodeConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rng: np.random.Generator,
) -> _ActionGenerator:
    if isinstance(config, ExpertGuidedEpisodeConfig):
        return ExpertGuidedActionGenerator(config, action_low, action_high, rng)
    if isinstance(config, PerturbRecoverEpisodeConfig):
        return PerturbRecoverActionGenerator(config, action_low, action_high, rng)
    if isinstance(config, SmoothRandomEpisodeConfig):
        return SmoothRandomActionGenerator(config, action_low, action_high, rng)
    raise TypeError(f"Unsupported episode configuration: {type(config).__name__}.")
