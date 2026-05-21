from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

from battlefield_rl.config import (
    AgentConfig,
    CurriculumConfig,
    CurriculumThreshold,
    EnvConfig,
    PlannerConfig,
    TrainingSceneConfig,
)
from battlefield_rl.env import EnemySpec, TacticalBattlefieldEnv
from battlefield_rl.map import BattlefieldMap, LocalSceneSampler
from battlefield_rl.planner import plan_global_path
from battlefield_rl.rl import D3QNAgent, Transition
from battlefield_rl.train.curriculum import StageGate

Coord = Tuple[int, int]


@dataclass
class EpisodeStats:
    episode: int
    stage: int
    steps: int
    reward: float
    reached_goal: bool
    timeout: bool
    path_efficiency: float
    rolling_success_rate: float
    rolling_timeout_rate: float
    rolling_path_efficiency: float
    guide_prob: float
    guided_actions: int
    eval_ran: bool = False
    source: str = ""
    enemy_case: str = ""


class HierarchicalTrainer:
    def __init__(
        self,
        grid: BattlefieldMap,
        env_cfg: EnvConfig,
        planner_cfg: PlannerConfig,
        agent_cfg: AgentConfig,
        curriculum_cfg: CurriculumConfig,
        gate_cfg: CurriculumThreshold,
        scene_cfg: TrainingSceneConfig | None = None,
        device: str = "cpu",
    ):
        self.grid = grid
        self.env_cfg = env_cfg
        self.planner_cfg = planner_cfg
        self.agent = D3QNAgent(agent_cfg, device=device)
        self.curriculum_cfg = curriculum_cfg
        self.gate_cfg = gate_cfg
        self.scene_cfg = scene_cfg or TrainingSceneConfig(enabled=False)
        self.scene_sampler = (
            LocalSceneSampler(grid, self.scene_cfg, env_cfg.window_size) if self.scene_cfg.enabled else None
        )

        self.stage = 1
        self.stage_episode = 0
        self.last_eval_metrics = None
        self.gate = StageGate(window_size=gate_cfg.window_size)

    def guide_probability(self) -> float:
        if not self.curriculum_cfg.heuristic_guidance:
            return 0.0
        if self.stage == 1:
            start = self.curriculum_cfg.stage1_guide_start
            end = self.curriculum_cfg.stage1_guide_end
            decay = self.curriculum_cfg.stage1_guide_decay_episodes
        elif self.stage == 2:
            start = self.curriculum_cfg.stage2_guide_start
            end = self.curriculum_cfg.stage2_guide_end
            decay = self.curriculum_cfg.stage2_guide_decay_episodes
        else:
            start = self.curriculum_cfg.stage3_guide_start
            end = self.curriculum_cfg.stage3_guide_end
            decay = self.curriculum_cfg.stage3_guide_decay_episodes

        ratio = min(1.0, self.stage_episode / max(1, decay))
        return start + (end - start) * ratio

    def _threat_scale(self) -> float:
        if self.stage == 1:
            return self.curriculum_cfg.threat_scale_stage1
        if self.stage == 2:
            return self.curriculum_cfg.threat_scale_stage2
        return self.curriculum_cfg.threat_scale_stage3

    def _enemy_count_range(self) -> Tuple[int, int]:
        if self.stage == 2:
            return self.curriculum_cfg.stage2_enemy_count_min, self.curriculum_cfg.stage2_enemy_count_max
        return self.curriculum_cfg.stage3_enemy_count_min, self.curriculum_cfg.stage3_enemy_count_max

    def _inner_ratio(self) -> float:
        if self.stage == 2:
            return self.curriculum_cfg.stage2_inner_ratio
        return self.curriculum_cfg.stage3_inner_ratio

    def _sample_fov_and_range(self) -> Tuple[float, int]:
        fov = random.uniform(self.curriculum_cfg.enemy_fov_min_deg, self.curriculum_cfg.enemy_fov_max_deg)
        rng = random.randint(self.curriculum_cfg.enemy_range_min, self.curriculum_cfg.enemy_range_max)
        return fov, rng

    def _sample_inner_enemy(self, center: Coord) -> EnemySpec:
        half = self.env_cfg.window_size // 2
        rows, cols = self.grid.shape
        r = random.randint(max(0, center[0] - half), min(rows - 1, center[0] + half))
        c = random.randint(max(0, center[1] - half), min(cols - 1, center[1] + half))
        fov, rng = self._sample_fov_and_range()
        return EnemySpec(row=r, col=c, facing_deg=random.uniform(0.0, 360.0), fov_deg=fov, max_range=rng)

    def _sample_outer_enemy(self, center: Coord) -> Optional[EnemySpec]:
        rows, cols = self.grid.shape
        half = self.env_cfg.window_size // 2
        band = self.curriculum_cfg.outer_band_extra

        inner_r_min = max(0, center[0] - half)
        inner_r_max = min(rows - 1, center[0] + half)
        inner_c_min = max(0, center[1] - half)
        inner_c_max = min(cols - 1, center[1] + half)

        outer_r_min = max(0, center[0] - half - band)
        outer_r_max = min(rows - 1, center[0] + half + band)
        outer_c_min = max(0, center[1] - half - band)
        outer_c_max = min(cols - 1, center[1] + half + band)

        candidates: List[Coord] = []
        for r in range(outer_r_min, outer_r_max + 1):
            for c in range(outer_c_min, outer_c_max + 1):
                in_inner = inner_r_min <= r <= inner_r_max and inner_c_min <= c <= inner_c_max
                if not in_inner:
                    candidates.append((r, c))

        if not candidates:
            return None

        r, c = random.choice(candidates)
        fov, rng = self._sample_fov_and_range()

        # 让外圈敌人更可能影响窗口：视距至少覆盖到窗口边缘距离
        dr = 0 if inner_r_min <= r <= inner_r_max else min(abs(r - inner_r_min), abs(r - inner_r_max))
        dc = 0 if inner_c_min <= c <= inner_c_max else min(abs(c - inner_c_min), abs(c - inner_c_max))
        min_dist = int((dr * dr + dc * dc) ** 0.5)
        rng = max(rng, min_dist + 5)

        return EnemySpec(row=r, col=c, facing_deg=random.uniform(0.0, 360.0), fov_deg=fov, max_range=rng)

    def _sample_enemies(self, center: Coord) -> List[EnemySpec]:
        if self.stage == 1:
            return []

        cmin, cmax = self._enemy_count_range()
        n = random.randint(cmin, cmax)
        inner_ratio = self._inner_ratio()

        enemies: List[EnemySpec] = []
        for _ in range(n):
            if random.random() < inner_ratio:
                enemies.append(self._sample_inner_enemy(center))
            else:
                outer_enemy = self._sample_outer_enemy(center)
                if outer_enemy is not None:
                    enemies.append(outer_enemy)
                else:
                    enemies.append(self._sample_inner_enemy(center))
        return enemies

    def _maybe_advance_stage(self) -> None:
        if self.stage >= self.curriculum_cfg.max_stage:
            return
        if not self.gate.ready():
            return
        metrics = self.gate.summary()
        min_success, max_timeout, max_eff = self.gate_cfg.for_stage(self.stage)
        if (
            metrics.success_rate >= min_success
            and metrics.timeout_rate <= max_timeout
            and metrics.path_efficiency <= max_eff
        ):
            self.stage += 1
            self.stage_episode = 0
            self.gate = StageGate(window_size=self.gate_cfg.window_size)

    def _run_single_episode(
        self,
        episode_idx: int,
        start: Coord,
        goal: Coord,
        train: bool,
        use_guidance: bool,
        epsilon_override: float | None = None,
        update_stage_gate: bool = False,
    ) -> EpisodeStats:
        source = "fullmap"
        enemy_case = "sampled"
        if self.scene_sampler is not None:
            scene = self.scene_sampler.sample(stage=self.stage)
            grid = scene.grid
            start = scene.start
            goal = scene.goal
            enemies = scene.enemies
            source = scene.source
            enemy_case = scene.enemy_case
        else:
            grid = self.grid
            enemies = self._sample_enemies(center=start)

        plan = plan_global_path(
            grid,
            start,
            goal,
            max_height_diff=self.env_cfg.max_height_diff,
            waypoint_interval=self.planner_cfg.waypoint_interval,
            allow_diagonal=self.planner_cfg.allow_diagonal,
        )
        if not plan.path:
            return EpisodeStats(
                episode_idx,
                self.stage,
                0,
                -100.0,
                False,
                True,
                999.0,
                0.0,
                1.0,
                999.0,
                0.0,
                0,
                False,
                source,
                enemy_case,
            )

        env = TacticalBattlefieldEnv(grid, self.env_cfg)
        obs = env.reset(
            start,
            goal,
            enemies=enemies,
            waypoints=[goal],
            threat_scale=self._threat_scale(),
        )

        total_reward = 0.0
        timeout = False
        guide_prob = self.guide_probability()
        guided_actions = 0

        for t in range(self.env_cfg.timeout_steps):
            mask = env.action_mask()
            expert_action = env.heuristic_action(
                distance_weight=self.curriculum_cfg.heuristic_distance_weight,
                threat_weight=self.curriculum_cfg.heuristic_threat_weight,
                revisit_weight=self.curriculum_cfg.heuristic_revisit_weight,
            )
            guided = False
            if train and use_guidance and random.random() < guide_prob:
                action = expert_action
                guided_actions += 1
                guided = True
            else:
                action = self.agent.act(obs, mask, epsilon_override=epsilon_override)
            step = env.step(action)
            next_obs = step.observation
            next_mask = env.action_mask()

            if train:
                tr = Transition(
                    state=obs,
                    action=action,
                    reward=step.reward,
                    next_state=next_obs,
                    done=step.done,
                    mask=mask,
                    next_mask=next_mask,
                    guided=guided,
                    expert_action=expert_action if train else -1,
                )
                self.agent.push_transition(tr)
                self.agent.learn_step()
                self.agent.global_step += 1
                self.agent.maybe_sync_target()

            total_reward += step.reward
            obs = next_obs

            if bool(step.info.get("timeout", False)):
                timeout = True

            if env.should_replan() and not step.done:
                break
            if step.done:
                break

        reached_goal = env.pos == goal
        steps = t + 1
        shortest_steps = max(1, len(plan.path) - 1)
        path_eff = max(1.0, steps / shortest_steps)

        if train:
            self.stage_episode += 1

        if update_stage_gate:
            self.gate.update(reached_goal=reached_goal, timeout=timeout, path_efficiency=path_eff)
            metrics = self.gate.summary()
        else:
            metrics = self.gate.summary()

        return EpisodeStats(
            episode=episode_idx,
            stage=self.stage,
            steps=steps,
            reward=float(total_reward),
            reached_goal=reached_goal,
            timeout=timeout,
            path_efficiency=float(path_eff),
            rolling_success_rate=metrics.success_rate,
            rolling_timeout_rate=metrics.timeout_rate,
            rolling_path_efficiency=metrics.path_efficiency,
            guide_prob=guide_prob,
            guided_actions=guided_actions,
            eval_ran=False,
            source=source,
            enemy_case=enemy_case,
        )

    def run_episode(self, episode_idx: int, start: Coord, goal: Coord, train: bool = True) -> EpisodeStats:
        stats = self._run_single_episode(
            episode_idx=episode_idx,
            start=start,
            goal=goal,
            train=train,
            use_guidance=True,
            epsilon_override=None,
            update_stage_gate=False,
        )
        if train and episode_idx % self.gate_cfg.eval_every_episodes == 0:
            self.evaluate_for_stage_gate(start, goal)
            metrics = self.gate.summary()
            stats.stage = self.stage
            stats.rolling_success_rate = metrics.success_rate
            stats.rolling_timeout_rate = metrics.timeout_rate
            stats.rolling_path_efficiency = metrics.path_efficiency
            stats.eval_ran = True
        return stats

    def evaluate_for_stage_gate(self, start: Coord, goal: Coord) -> None:
        for _ in range(self.gate_cfg.eval_episodes):
            self._run_single_episode(
                episode_idx=0,
                start=start,
                goal=goal,
                train=False,
                use_guidance=False,
                epsilon_override=self.gate_cfg.eval_epsilon,
                update_stage_gate=True,
            )
        self.last_eval_metrics = self.gate.summary()
        self._maybe_advance_stage()
