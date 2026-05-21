from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 25
    max_height_diff: int = 0
    los_height_margin: float = 0.0

    step_penalty: float = -0.01
    progress_reward_scale: float = 0.3
    exposure_penalty_scale: float = 0
    visible_event_penalty: float = 0
    revisit_penalty_scale: float = 0

    waypoint_reward: float = 1.0
    goal_reward: float = 50.0
    timeout_penalty: float = -60.0

    timeout_steps: int = 50
    no_progress_limit: int = 10


@dataclass(frozen=True)
class PlannerConfig:
    waypoint_interval: int = 22
    allow_diagonal: bool = True


@dataclass(frozen=True)
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150_000
    target_sync_steps: int = 2_000
    batch_size: int = 64
    memory_size: int = 200_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 250_000


@dataclass(frozen=True)
class CurriculumThreshold:
    window_size: int = 200
    eval_every_episodes: int = 50
    eval_episodes: int = 50
    eval_epsilon: float = 0.02
    stage1_success_rate: float = 0.85
    stage1_timeout_rate: float = 0.15
    stage1_path_efficiency: float = 1.80
    stage2_success_rate: float = 0.80
    stage2_timeout_rate: float = 0.20
    stage2_path_efficiency: float = 2.00

    def for_stage(self, stage: int) -> tuple[float, float, float]:
        if stage <= 1:
            return self.stage1_success_rate, self.stage1_timeout_rate, self.stage1_path_efficiency
        return self.stage2_success_rate, self.stage2_timeout_rate, self.stage2_path_efficiency


@dataclass(frozen=True)
class CurriculumConfig:
    max_stage: int = 3
    threat_scale_stage1: float = 0.0
    threat_scale_stage2: float = 0.5
    threat_scale_stage3: float = 1.0

    stage2_enemy_count_min: int = 1
    stage2_enemy_count_max: int = 1
    stage3_enemy_count_min: int = 1
    stage3_enemy_count_max: int = 3

    stage2_inner_ratio: float = 0.8
    stage3_inner_ratio: float = 0.5

    enemy_fov_min_deg: float = 90.0
    enemy_fov_max_deg: float = 90.0
    enemy_range_min: int = 18
    enemy_range_max: int = 18

    outer_band_extra: int = 5

    heuristic_guidance: bool = True
    stage1_guide_start: float = 0.80
    stage1_guide_end: float = 0.10
    stage1_guide_decay_episodes: int = 1000
    stage2_guide_start: float = 0.50
    stage2_guide_end: float = 0.05
    stage2_guide_decay_episodes: int = 1200
    stage3_guide_start: float = 0.20
    stage3_guide_end: float = 0.00
    stage3_guide_decay_episodes: int = 1500

    heuristic_distance_weight: float = 1.0
    heuristic_threat_weight: float = 1.5
    heuristic_revisit_weight: float = 0.2


@dataclass(frozen=True)
class TrainingSceneConfig:
    enabled: bool = True
    scene_size: int = 35
    procedural_ratio: float = 0.70

    enemy_fov_deg: float = 90.0
    enemy_range: int = 18
    enemy_facing_jitter_deg: float = 12.0

    stage2_inside_ratio: float = 0.70
    stage2_none_ratio: float = 0.30
    stage3_inside_ratio: float = 0.40
    stage3_outer_ratio: float = 0.40
    stage3_none_ratio: float = 0.20

    goal_min_distance: int = 8
    goal_max_distance: int = 22
    max_sample_attempts: int = 100
