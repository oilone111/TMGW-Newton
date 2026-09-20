from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .dataset import TrainingSample
from .network import torch

if torch is not None:
    from .network import TMGWNetwork
else:  # Allows dataset/normalization checks without installing the AI backend.
    TMGWNetwork = None


CHECKPOINT_FORMAT_VERSION = 11


def _moments(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return feature-wise mean/std without concatenating all graph arrays."""
    if not arrays:
        raise ValueError("normalization requires at least one array")
    count = 0
    total = np.zeros(arrays[0].shape[1], dtype=np.float64)
    total_sq = np.zeros_like(total)
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        count += values.shape[0]
        total += np.sum(values, axis=0)
        total_sq += np.sum(values * values, axis=0)
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _target_matrix(sample: TrainingSample) -> np.ndarray:
    n = sample.graph.reservoir_node_count
    nw = sample.graph.well_node_count
    correction = sample.target_correction
    target = np.zeros((n + nw, 4), dtype=np.float32)
    target[:n, 0] = correction[:n]
    target[:n, 1] = correction[n : 2 * n]
    target[:n, 2] = correction[2 * n : 3 * n]
    if nw:
        target[n : n + nw, 3] = correction[3 * n :]
    return target


def _normalization(
    samples: list[TrainingSample], model_kind: str,
) -> dict[str, list[float] | int | str]:
    node_arrays = [sample.graph.node_features for sample in samples]
    edge_arrays: list[np.ndarray] = []
    for sample in samples:
        features = sample.graph.edge_features.copy()
        if model_kind.upper() != "TMGW":
            features[:, 4:6] = 1.0
        edge_arrays.append(features)
    node_mean, node_scale = _moments(node_arrays)
    edge_mean, edge_scale = _moments(edge_arrays)

    sum_sq = np.zeros(4, dtype=np.float64)
    counts = np.zeros(4, dtype=np.float64)
    for sample in samples:
        target = _target_matrix(sample)
        n = sample.graph.reservoir_node_count
        sum_sq[:3] += np.sum(target[:n, :3] ** 2, axis=0)
        counts[:3] += n
        if sample.graph.well_node_count:
            sum_sq[3] += np.sum(target[n:, 3] ** 2)
            counts[3] += sample.graph.well_node_count
    output_scale = np.sqrt(sum_sq / np.maximum(counts, 1.0))
    output_scale = np.maximum(output_scale, np.asarray([1e-4, 2e-5, 1e-5, 1e-4]))
    return {
        "version": CHECKPOINT_FORMAT_VERSION,
        "source": "training_split_only",
        "node_mean": node_mean.tolist(),
        "node_scale": node_scale.tolist(),
        "edge_mean": edge_mean.tolist(),
        "edge_scale": edge_scale.tolist(),
        "output_scale": output_scale.astype(np.float32).tolist(),
    }


def _sample_difficulty(sample: TrainingSample, output_scale: np.ndarray) -> float:
    """Rank samples by the non-dimensional correction energy.

    The correction data are strongly heavy-tailed: most time steps are nearly
    linear while a small fraction control Newton cost.  A logarithm preserves
    the ranking without allowing a single step to dominate every epoch.
    """
    target = _target_matrix(sample)
    n = sample.graph.reservoir_node_count
    nw = sample.graph.well_node_count
    pressure = np.mean((target[:n, 0] / output_scale[0]) ** 2)
    water = np.mean((target[:n, 1] / output_scale[1]) ** 2)
    gas = np.mean((target[:n, 2] / output_scale[2]) ** 2)
    well = (
        np.mean((target[n : n + nw, 3] / output_scale[3]) ** 2)
        if nw else 0.0
    )
    correction_energy = np.log1p(pressure + water + gas + well)
    # A warm start only has practical value on steps that actually consume
    # nonlinear work.  Include the recorded S1/S0 difficulty so the sampler
    # cannot be dominated by the many nearly linear states.
    nonlinear_cost = max(float(sample.baseline_newton_iterations) - 2.0, 0.0)
    defect = np.log1p(max(float(sample.baseline_initial_residual), 0.0))
    return float(correction_energy + 0.45 * nonlinear_cost + 0.15 * defect)


def _select_epoch_samples(
    samples: list[TrainingSample],
    difficulties: np.ndarray,
    requested: int,
    hard_fraction: float,
) -> list[TrainingSample]:
    """Choose a stable mixture of difficult and ordinary time steps."""
    if requested <= 0 or requested >= len(samples):
        return list(samples)
    requested = max(1, requested)
    hard_fraction = float(np.clip(hard_fraction, 0.0, 0.9))
    hard_count = min(requested, int(round(requested * hard_fraction)))
    ranking = np.argsort(difficulties)
    hard_pool_size = min(len(samples), max(hard_count * 2, len(samples) // 4, 1))
    hard_pool = ranking[-hard_pool_size:].tolist()
    selected_indices = random.sample(hard_pool, min(hard_count, len(hard_pool)))
    selected_set = set(selected_indices)
    remainder = [index for index in range(len(samples)) if index not in selected_set]
    ordinary_count = requested - len(selected_indices)
    selected_indices.extend(random.sample(remainder, ordinary_count))
    random.shuffle(selected_indices)
    return [samples[index] for index in selected_indices]


def _fixed_validation_subset(
    samples: list[TrainingSample], difficulties: np.ndarray, requested: int,
) -> list[TrainingSample]:
    """Return a deterministic mix of hard and representative validation steps."""
    if requested <= 0 or requested >= len(samples):
        return list(samples)
    ranking = np.argsort(difficulties)
    hard_count = max(1, requested // 2)
    selected = ranking[-hard_count:].tolist()
    remainder_count = requested - hard_count
    if remainder_count:
        ordinary = ranking[: max(len(ranking) - hard_count, 1)]
        positions = np.linspace(0, len(ordinary) - 1, remainder_count).round().astype(int)
        selected.extend(ordinary[positions].tolist())
    return [samples[index] for index in selected]


def _tensor(value, device, dtype=None):
    return torch.as_tensor(value, device=device, dtype=dtype)


def train_network(
    train_samples: list[TrainingSample],
    validation_samples: list[TrainingSample],
    output_checkpoint: str | Path,
    epochs: int = 300,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    patience: int = 30,
    batch_size: int = 4,
    samples_per_epoch: int = 0,
    validation_interval: int = 5,
    seed: int = 20260818,
    device: str = "cpu",
    model_kind: str = "TMGW",
    initial_checkpoint: str | Path | None = None,
    state_weight: float = 0.0,
    residual_weight: float = 0.0,
    direction_weight: float = 0.0,
    mass_weight: float = 0.0,
    bound_weight: float = 0.0,
    active_node_boost: float = 4.0,
    hard_sample_fraction: float = 0.55,
    amplitude_weight: float = 0.0,
    hard_step_boost: float = 1.0,
    target_projection: float = 0.50,
    curriculum_epochs: int = 1,
    online_validation_samples: int = 32,
    online_validation_interval: int = 10,
    relaxation_weight: float = 1.0,
) -> dict[str, list[float]]:
    if torch is None:
        raise ImportError("PyTorch is required for network training")
    assert TMGWNetwork is not None
    if not train_samples:
        raise ValueError("training sample list is empty")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_cfg = {
        "node_dim": train_samples[0].graph.node_features.shape[1],
        "edge_dim": train_samples[0].graph.edge_features.shape[1],
        "hidden_dim": 24,
        "layers": 1,
        "n_relations": 1 if model_kind.upper() == "GNN" else 6,
        "use_threshold_gate": model_kind.upper() == "TMGW",
        "output_mode": "global_relaxation_v11_compact",
    }
    normalization = _normalization(train_samples, model_kind)
    training_settings = {
        "state_weight": state_weight,
        "residual_weight": residual_weight,
        "direction_weight": direction_weight,
        "mass_weight": mass_weight,
        "bound_weight": bound_weight,
        "active_node_boost": active_node_boost,
        "hard_sample_fraction": hard_sample_fraction,
        "amplitude_weight": amplitude_weight,
        "hard_step_boost": hard_step_boost,
        "target_projection": target_projection,
        "curriculum_epochs": curriculum_epochs,
        "online_validation_samples": online_validation_samples,
        "online_validation_interval": online_validation_interval,
        "relaxation_weight": relaxation_weight,
    }
    network = TMGWNetwork(**model_cfg).to(device)
    resume_best_loss = float("inf")
    if initial_checkpoint is not None:
        payload = torch.load(Path(initial_checkpoint), map_location=device, weights_only=True)
        if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "resume checkpoint is not a v0.18 compact model; start a fresh v0.18 training run"
            )
        if payload.get("model_config") != model_cfg:
            raise ValueError("resume checkpoint architecture does not match the requested model")
        saved_normalization = payload.get("normalization")
        if not isinstance(saved_normalization, dict):
            raise ValueError("resume checkpoint has no normalization statistics")
        for key in ("node_mean", "node_scale", "edge_mean", "edge_scale", "output_scale"):
            if not np.allclose(saved_normalization[key], normalization[key], rtol=1e-5, atol=1e-7):
                raise ValueError(f"resume checkpoint normalization mismatch: {key}")
        network.load_state_dict(payload["state_dict"])
        # Preserve the best complete-residual score across parameter changes.
        # Resetting it to infinity would let a merely different run be labelled
        # "learned" without actually beating the matched P0 baseline.
        resume_best_loss = float(payload.get("best_online_score", float("inf")))
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1.0e-5
    )
    history = {
        "train": [], "validation": [], "learning_rate": [],
        "train_state": [], "train_residual": [], "train_direction": [], "train_mass": [],
        "train_amplitude": [], "train_relaxation": [],
        "validation_state": [], "validation_residual": [],
        "validation_direction": [], "validation_mass": [], "validation_amplitude": [],
        "validation_relaxation": [],
        "selected_difficulty": [],
        "validation_true_residual_ratio": [],
        "validation_true_residual_q75": [],
        "validation_true_improvement_rate": [],
        "validation_true_fallback_rate": [],
        "validation_inference_gain": [],
        "validation_selected_alpha_median": [],
        "validation_direct_acceptance_rate": [],
        "validation_deep_improvement_rate": [],
        "validation_online_score": [],
    }
    output_checkpoint = Path(output_checkpoint)
    partial_history_path = output_checkpoint.with_suffix(".partial.history.json")
    final_history_path = output_checkpoint.with_suffix(".history.json")
    prior_history: dict[str, list[float]] = {}
    if initial_checkpoint is not None:
        history_source = partial_history_path if partial_history_path.exists() else final_history_path
        if history_source.exists():
            try:
                loaded_history = json.loads(history_source.read_text(encoding="utf-8"))
                prior_history = {
                    key: [float(value) for value in values]
                    for key, values in loaded_history.items()
                    if isinstance(values, list)
                }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                prior_history = {}
    best_online_score = resume_best_loss
    stale = 0

    node_mean = _tensor(normalization["node_mean"], device)
    node_scale = _tensor(normalization["node_scale"], device)
    edge_mean = _tensor(normalization["edge_mean"], device)
    edge_scale = _tensor(normalization["edge_scale"], device)
    output_scale = _tensor(normalization["output_scale"], device)
    difficulty_values = np.asarray([
        _sample_difficulty(sample, np.asarray(normalization["output_scale"], dtype=float))
        for sample in train_samples
    ])
    validation_difficulty = np.asarray([
        _sample_difficulty(sample, np.asarray(normalization["output_scale"], dtype=float))
        for sample in validation_samples
    ])
    online_samples = _fixed_validation_subset(
        validation_samples, validation_difficulty, online_validation_samples
    )

    def predict_matrix(sample: TrainingSample):
        graph = sample.graph
        edge_features_np = graph.edge_features.copy()
        if model_kind.upper() != "TMGW":
            edge_features_np[:, 4:6] = 1.0
        pred_scaled = network(
            (_tensor(graph.node_features, device) - node_mean) / node_scale,
            _tensor(graph.edge_index, device, torch.long),
            (_tensor(edge_features_np, device) - edge_mean) / edge_scale,
            _tensor(graph.edge_relation, device, torch.long),
        )
        if model_cfg["output_mode"] == "global_relaxation_v11_compact":
            physics_defect = _tensor(graph.node_features[:, 26:30], device)
            pred = pred_scaled.reshape(1, 1) * physics_defect
        elif model_cfg["output_mode"] in {"physics_multiplier_v7", "relaxation_teacher_v8"}:
            # The graph stores a diagonal, physics-derived defect correction
            # in columns 26:30.  TMGW learns spatially varying relaxation
            # factors, turning a difficult full-field regression into a
            # bounded learned preconditioner.
            physics_defect = _tensor(graph.node_features[:, 26:30], device)
            pred = pred_scaled * physics_defect
        else:
            pred = pred_scaled * output_scale
        n = graph.reservoir_node_count
        nw = graph.well_node_count
        target = _tensor(_target_matrix(sample), device)
        gas_active = (torch.max(torch.abs(target[:n, 2])) > 1.0e-12).to(pred.dtype)
        reservoir_pred = torch.cat([
            torch.stack([pred[:n, 0], pred[:n, 1], pred[:n, 2] * gas_active], dim=-1),
            pred[:n, 3:4],
        ], dim=-1)
        well_pred = pred[n:]
        if nw:
            well_pred = torch.stack([
                well_pred[:, 0], well_pred[:, 1], well_pred[:, 2],
                well_pred[:, 3] * _tensor(graph.node_features[n:, 20], device),
            ], dim=-1)
        return (
            torch.cat([reservoir_pred, well_pred], dim=0), pred_scaled,
            edge_features_np, target, gas_active,
        )

    def sample_loss(sample: TrainingSample, phase: float = 1.0):
        graph = sample.graph
        pred, pred_scaled, edge_features_np, target, gas_active = predict_matrix(sample)
        n = graph.reservoir_node_count
        nw = graph.well_node_count
        pore_weight = torch.pow(10.0, _tensor(graph.node_features[:n, 19], device))
        pore_weight = pore_weight / torch.sum(pore_weight)
        def active_component_loss(predicted, expected, scale):
            normalized_target = torch.abs(expected / scale)
            activity = torch.pow(torch.clamp(normalized_target, 0.0, 4.0), 0.35)
            weights = 1.0 + active_node_boost * activity
            weights = weights / torch.clamp(torch.mean(weights), min=1.0e-6)
            error = (predicted - expected) / scale
            robust = torch.nn.functional.smooth_l1_loss(
                error, torch.zeros_like(error), beta=0.25, reduction="none"
            )
            return torch.mean(weights * robust)

        pressure_loss = active_component_loss(pred[:n, 0], target[:n, 0], output_scale[0])
        water_loss = active_component_loss(pred[:n, 1], target[:n, 1], output_scale[1])
        gas_loss = active_component_loss(pred[:n, 2], target[:n, 2], output_scale[2])
        component_count = 2.0 + gas_active
        reservoir_loss = (
            pressure_loss + water_loss + gas_active * gas_loss
        ) / component_count
        well_loss = (
            active_component_loss(pred[n:, 3], target[n:, 3], output_scale[3])
            if nw else torch.zeros((), device=device)
        )
        state_loss = (2.0 * reservoir_loss + well_loss) / (3.0 if nw else 2.0)

        # Direction matters more than exact amplitude for a Newton warm start.
        active_pred = torch.cat([
            pred[:n, 0] / output_scale[0],
            pred[:n, 1] / output_scale[1],
            pred[:n, 2] * gas_active / output_scale[2],
            pred[n:, 3] / output_scale[3] if nw else pred.new_empty(0),
        ])
        active_target = torch.cat([
            target[:n, 0] / output_scale[0],
            target[:n, 1] / output_scale[1],
            target[:n, 2] * gas_active / output_scale[2],
            target[n:, 3] / output_scale[3] if nw else target.new_empty(0),
        ])
        pred_norm = torch.sqrt(torch.sum(active_pred ** 2) + 1.0e-4)
        target_norm = torch.sqrt(torch.sum(active_target ** 2) + 1.0e-8)
        cosine = torch.sum(active_pred * active_target) / (pred_norm * target_norm)
        direction_loss = 1.0 - torch.clamp(cosine, -1.0, 1.0)

        # v0.12 often learned the correct sign but with an amplitude too small
        # to cross a Newton stopping boundary.  Penalize near-zero collapse by
        # requiring the predicted correction to cover a configurable fraction
        # of the converged-state direction, while softly limiting overshoot.
        target_energy = torch.sum(active_target ** 2)
        projection = torch.sum(active_pred * active_target) / (target_energy + 1.0e-6)
        target_active = (target_energy > 1.0e-4).to(pred.dtype)
        amplitude_loss = target_active * (
            torch.relu(target_projection - projection) ** 2
            + 0.10 * torch.relu(projection - 1.50) ** 2
        )

        # v0.18 predicts one coherent relaxation for the complete time-step
        # defect.  Its teacher selects the largest safely contractive step,
        # rather than the residual minimum, so training is aligned with the
        # observed Newton-iteration boundary while preserving direction.
        relaxation_prediction = pred_scaled.reshape(-1)[0]
        relaxation_target = torch.as_tensor(
            float(sample.target_relaxation), dtype=pred.dtype, device=device
        )
        relaxation_loss = torch.nn.functional.smooth_l1_loss(
            relaxation_prediction, relaxation_target, beta=0.05,
        )

        # Linearized graph-flux residual relative to the converged target.
        src, dst = _tensor(graph.edge_index, device, torch.long)
        gate = _tensor(edge_features_np[:, 4], device)
        if not model_cfg["use_threshold_gate"]:
            gate = torch.ones_like(gate)
        conductance_log = _tensor(graph.edge_features[:, 0], device)
        normalized_log = torch.clamp(conductance_log - torch.max(conductance_log), -12.0, 0.0)
        conductance = torch.pow(10.0, normalized_log)
        pred_pressure = torch.cat([pred[:n, 0], pred[n:, 3]])
        target_pressure = torch.cat([target[:n, 0], target[n:, 3]])
        q_pred = conductance * gate * (pred_pressure[src] - pred_pressure[dst])
        q_target = conductance * gate * (target_pressure[src] - target_pressure[dst])
        div_pred = torch.zeros(pred.shape[0], device=device)
        div_target = torch.zeros(pred.shape[0], device=device)
        div_pred.index_add_(0, dst, q_pred)
        div_target.index_add_(0, dst, q_target)
        divergence_scale = torch.maximum(
            torch.sqrt(torch.mean(div_target ** 2)), torch.tensor(1.0e-5, device=device)
        )
        residual_loss = torch.mean(((div_pred - div_target) / divergence_scale) ** 2)

        # Pore-volume-weighted water-inventory correction.
        water_mass = (
            torch.sum(pore_weight * pred[:n, 1])
            - torch.sum(pore_weight * target[:n, 1])
        ) ** 2
        gas_mass = (
            torch.sum(pore_weight * pred[:n, 2])
            - torch.sum(pore_weight * target[:n, 2])
        ) ** 2
        mass_loss = water_mass / output_scale[1] ** 2 + gas_mass / output_scale[2] ** 2
        bound_loss = torch.mean(
            torch.relu(torch.abs(pred[:n, 1:3]) - 0.1) ** 2
        )
        ramp = float(np.clip(phase, 0.0, 1.0))
        hard_step_factor = 1.0 + hard_step_boost * min(
            max((float(sample.baseline_newton_iterations) - 3.0) / 3.0, 0.0), 1.0
        )
        if str(model_cfg.get("output_mode", "")) == "global_relaxation_v11_compact":
            # The compact model optimizes only the graph-level relaxation
            # supplied by the Newton-margin teacher. Proxy diagnostics are
            # recorded but excluded from the objective so unrelated NaNs
            # cannot contaminate the valid training gradient.
            total_loss = hard_step_factor * relaxation_weight * relaxation_loss
        else:
            total_loss = hard_step_factor * (
                state_weight * state_loss
                + residual_weight * ramp * residual_loss
                + direction_weight * (0.25 + 0.75 * ramp) * direction_loss
                + amplitude_weight * (0.35 + 0.65 * ramp) * amplitude_loss
                + mass_weight * ramp * mass_loss
                + bound_weight * bound_loss
                + relaxation_weight * relaxation_loss
            )
        return total_loss, torch.stack([
            state_loss, residual_loss, direction_loss, mass_loss, amplitude_loss,
            relaxation_loss,
        ])

    def true_residual_metrics(
        gains: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluate the learned scalar with the v0.18 compact single-candidate gate."""
        if gains is None:
            gains = np.asarray([1.0], dtype=float)
        selected_ratios: list[float] = []
        selected_gains: list[float] = []
        network.eval()
        with torch.no_grad():
            for sample in online_samples:
                model = sample.model
                current = sample.current_state
                if model is None or current is None:
                    continue
                prediction, relaxation_scaled, _, _, _ = predict_matrix(sample)
                predicted_relaxation = float(
                    relaxation_scaled.reshape(-1)[0].detach().cpu()
                )
                matrix = prediction.detach().cpu().numpy()
                n = sample.graph.reservoir_node_count
                nw = sample.graph.well_node_count
                correction = np.concatenate([
                    matrix[:n, 0], matrix[:n, 1], matrix[:n, 2], matrix[n : n + nw, 3],
                ])
                x_ext = sample.extrapolated_state
                r_ext = np.linalg.norm(
                    model.residual(x_ext, current, sample.dt_s), ord=np.inf
                )
                mass_ext = model.global_mass_error(model.unpack(x_ext), current, sample.dt_s)
                candidates: list[tuple[float, float]] = []
                for gain in gains:
                    candidate = model.warm_start_well_projection(x_ext + gain * correction)
                    if not np.all(np.isfinite(candidate)):
                        continue
                    residual = np.linalg.norm(
                        model.residual(candidate, current, sample.dt_s), ord=np.inf
                    )
                    mass = model.global_mass_error(model.unpack(candidate), current, sample.dt_s)
                    ratio = float(residual / max(r_ext, 1.0e-30))
                    mass_ratio = float(mass / max(
                        mass_ext, model.cfg.solver.warm_start_mass_tolerance
                    ))
                    if mass_ratio <= model.cfg.solver.residual_reject:
                        candidates.append((ratio, float(gain)))
                ratio, gain = min(candidates, default=(1.0, 0.0), key=lambda item: item[0])
                # A non-improving candidate is equivalent to the exact S1
                # fallback and must not be counted as AI use.
                if ratio >= 1.0:
                    ratio, gain = 1.0, 0.0
                selected_ratios.append(ratio)
                selected_gains.append(
                    predicted_relaxation * gain if gain > 0.0 else 0.0
                )
        if not selected_ratios:
            return {
                "median_ratio": float("inf"), "q75_ratio": float("inf"),
                "improvement_rate": 0.0, "fallback_rate": 1.0,
                "gain": 1.0, "median_alpha": 0.0,
                "direct_acceptance_rate": 0.0, "deep_improvement_rate": 0.0,
                "score": float("inf"),
            }
        ratio_array = np.asarray(selected_ratios, dtype=float)
        gain_array = np.asarray(selected_gains, dtype=float)
        median = float(np.median(ratio_array))
        q75 = float(np.quantile(ratio_array, 0.75))
        improvement_rate = float(np.mean(ratio_array < 0.90))
        deep_improvement_rate = float(np.mean(ratio_array < 0.50))
        fallback = float(np.mean(gain_array == 0.0))
        direct = float(np.mean(ratio_array <= 0.80))
        active_gain = gain_array[gain_array > 0.0]
        median_alpha = float(np.median(active_gain)) if active_gain.size else 0.0
        score = (
            median + 0.75 * q75 + 0.75 * fallback
            + 0.35 * (1.0 - improvement_rate)
            + 0.65 * (1.0 - deep_improvement_rate)
        )
        return {
            "score": score,
            "median_ratio": median,
            "q75_ratio": q75,
            "improvement_rate": improvement_rate,
            "fallback_rate": fallback,
            "gain": 1.0,
            "median_alpha": median_alpha,
            "direct_acceptance_rate": direct,
            "deep_improvement_rate": deep_improvement_rate,
        }

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the zero-decoder graph as an explicit untrained baseline. It
    # outputs the same global 0.5 used by P0.  v0.18 checkpoint selection is
    # driven by held-out Newton-margin labels, with the true residual gate kept
    # as a safety penalty instead of incorrectly forcing the residual minimum.
    if initial_checkpoint is None:
        network.eval()
        with torch.no_grad():
            baseline_validation_values = [
                float(sample_loss(sample, phase=1.0)[0].cpu())
                for sample in validation_samples
            ]
        baseline_online = true_residual_metrics()
        baseline_validation = float(np.mean(baseline_validation_values))
        best_online_score = (
            baseline_validation + 0.25 * float(baseline_online["fallback_rate"])
        )
        torch.save({
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": network.state_dict(),
            "model_config": model_cfg,
            "normalization": normalization,
            "best_validation_loss": baseline_validation,
            "best_online_score": best_online_score,
            "inference_gain": 1.0,
            "scale_strategy": "untrained_v18_compact_triggered_single_check",
            "online_validation_metrics": baseline_online,
            "seed": seed,
            "model_kind": model_kind.upper(),
            "training_settings": training_settings,
            "checkpoint_role": "untrained_v18_compact_baseline",
        }, output_checkpoint)
    for epoch in range(epochs):
        phase = min(1.0, (epoch + 1) / max(curriculum_epochs, 1))
        epoch_samples = _select_epoch_samples(
            train_samples, difficulty_values, samples_per_epoch, hard_sample_fraction
        )
        network.train()
        total = 0.0
        component_total = np.zeros(6, dtype=float)
        for batch_start in range(0, len(epoch_samples), batch_size):
            batch = epoch_samples[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            for sample in batch:
                loss, components = sample_loss(sample, phase=phase)
                (loss / len(batch)).backward()
                total += float(loss.detach().cpu())
                component_total += components.detach().cpu().numpy()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        train_loss = total / len(epoch_samples)
        history["train"].append(train_loss)
        selected_ids = {id(sample) for sample in epoch_samples}
        selected_difficulty = [
            difficulty_values[index] for index, sample in enumerate(train_samples)
            if id(sample) in selected_ids
        ]
        history["selected_difficulty"].append(float(np.mean(selected_difficulty)))
        history["learning_rate"].append(float(scheduler.get_last_lr()[0]))
        component_mean = component_total / len(epoch_samples)
        for key, value in zip(
            (
                "train_state", "train_residual", "train_direction", "train_mass",
                "train_amplitude", "train_relaxation",
            ),
            component_mean,
        ):
            history[key].append(float(value))

        evaluate_validation = epoch == 0 or (epoch + 1) % max(validation_interval, 1) == 0
        if evaluate_validation:
            network.eval()
            with torch.no_grad():
                evaluated = [sample_loss(s, phase=1.0) for s in validation_samples]
                values = [float(item[0].cpu()) for item in evaluated]
                validation_components = np.mean([
                    item[1].cpu().numpy() for item in evaluated
                ], axis=0)
            val_loss = float(np.mean(values)) if values else train_loss
        else:
            val_loss = history["validation"][-1] if history["validation"] else train_loss
            validation_components = np.asarray([
                history[key][-1] if history[key] else component_mean[index]
                for index, key in enumerate((
                    "validation_state", "validation_residual",
                    "validation_direction", "validation_mass", "validation_amplitude",
                    "validation_relaxation",
                ))
            ])
        history["validation"].append(val_loss)
        for key, value in zip(
            (
                "validation_state", "validation_residual", "validation_direction",
                "validation_mass", "validation_amplitude", "validation_relaxation",
            ),
            validation_components,
        ):
            history[key].append(float(value))

        evaluate_online = epoch == 0 or (epoch + 1) % max(online_validation_interval, 1) == 0
        if evaluate_online:
            online = true_residual_metrics()
        else:
            online = {
                "median_ratio": history["validation_true_residual_ratio"][-1],
                "q75_ratio": history["validation_true_residual_q75"][-1],
                "improvement_rate": history["validation_true_improvement_rate"][-1],
                "fallback_rate": history["validation_true_fallback_rate"][-1],
                "gain": history["validation_inference_gain"][-1],
                "median_alpha": history["validation_selected_alpha_median"][-1],
                "direct_acceptance_rate": history["validation_direct_acceptance_rate"][-1],
                "deep_improvement_rate": history["validation_deep_improvement_rate"][-1],
                "score": history["validation_online_score"][-1],
            }
        for key, value in (
            ("validation_true_residual_ratio", online["median_ratio"]),
            ("validation_true_residual_q75", online["q75_ratio"]),
            ("validation_true_improvement_rate", online["improvement_rate"]),
            ("validation_true_fallback_rate", online["fallback_rate"]),
            ("validation_inference_gain", online["gain"]),
            ("validation_selected_alpha_median", online["median_alpha"]),
            ("validation_direct_acceptance_rate", online["direct_acceptance_rate"]),
            ("validation_deep_improvement_rate", online["deep_improvement_rate"]),
            ("validation_online_score", online["score"]),
        ):
            history[key].append(float(value))

        # Compare against the P0-equivalent zero decoder using the held-out
        # Newton-margin objective.  A fallback penalty prevents a numerically
        # aggressive label fit from being saved when the physical gate rejects
        # too many candidates.
        selection_score = val_loss + 0.25 * float(online["fallback_rate"])
        if evaluate_validation and evaluate_online and selection_score < best_online_score:
            best_online_score = selection_score
            stale = 0
            torch.save({
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "state_dict": network.state_dict(),
                "model_config": model_cfg,
                "normalization": normalization,
                "best_validation_loss": val_loss,
                "best_online_score": best_online_score,
                "inference_gain": 1.0,
                "scale_strategy": "learned_v18_compact_triggered_single_check",
                "online_validation_metrics": online,
                "seed": seed,
                "model_kind": model_kind.upper(),
                "training_settings": training_settings,
                "checkpoint_role": "learned_tmgw_compact_newton_margin",
            }, output_checkpoint)
        elif evaluate_validation and evaluate_online:
            stale += 1
            if stale >= patience:
                print(
                    f"epoch={epoch + 1:04d} train={train_loss:.8e} "
                    f"validation={val_loss:.8e} early_stop=1",
                    flush=True,
                )
                break
        online_text = (
            f" true_ratio={online['median_ratio']:.6e} "
            f"true_q75={online['q75_ratio']:.6e} "
            f"alpha={online['median_alpha']:.5f} "
            f"fallback={online['fallback_rate']:.3f}"
            if evaluate_online else ""
        )
        print(
            f"epoch={epoch + 1:04d} train={train_loss:.8e} "
            f"validation={val_loss:.8e} lr={history['learning_rate'][-1]:.8e}"
            f"{online_text}",
            flush=True,
        )
        combined_history = {
            key: prior_history.get(key, []) + values for key, values in history.items()
        }
        partial_history_path.write_text(
            json.dumps(combined_history, indent=2), encoding="utf-8"
        )
    return {
        key: prior_history.get(key, []) + values for key, values in history.items()
    }
