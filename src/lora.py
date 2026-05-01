"""LoRA layers, handwritten truncated SVD initialization, and model utilities."""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


EPSILON = 1e-12
VALID_PRETRAINING_MODES = {"none", "standard", "truncated_svd"}


def safe_normalize(tensor: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """
    Normalize a tensor by its L2 norm with zero-safe handling.
    Args:
        tensor (torch.Tensor): Input tensor to normalize.
        eps (float): Minimum norm treated as nonzero.
    Returns:
        torch.Tensor: Normalized tensor or zeros when the norm is too small.
    Algorithm:
        1. Compute the tensor L2 norm.
        2. Return zeros if the norm is numerically zero.
        3. Divide the tensor by the norm otherwise.
    """
    norm = torch.norm(tensor)
    if norm <= eps:
        return torch.zeros_like(tensor)
    return tensor / norm


def truncated_svd_power_iteration(
    weight: torch.Tensor,
    rank: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a truncated SVD using handwritten power iteration and deflation.
    Args:
        weight (torch.Tensor): Two-dimensional weight matrix W.
        rank (int): Maximum number of singular components to compute.
        max_iter (int): Maximum power iterations per component.
        tol (float): Sign-invariant convergence tolerance.
        seed (Optional[int]): Optional random seed for deterministic vectors.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: U, S, and V where W ≈ U diag(S) V.T.
    Algorithm:
        1. Copy W into a stable floating-point residual matrix.
        2. Estimate each dominant right singular vector through R.T R power iteration.
        3. Compute sigma = ||R v|| and u = R v / sigma.
        4. Store u, sigma, v and deflate R by sigma * u * v.T.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(weight.shape)}")
    if rank <= 0:
        raise ValueError("rank must be positive")

    out_features, in_features = weight.shape
    rank = min(rank, out_features, in_features)
    original_dtype = weight.dtype
    work_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    residual = weight.detach().to(dtype=work_dtype).clone()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(seed)

    left_vectors = []
    singular_values = []
    right_vectors = []

    for _ in range(rank):
        vector = torch.randn(
            in_features,
            device=weight.device,
            dtype=work_dtype,
            generator=generator,
        )
        vector = safe_normalize(vector)
        previous_vector = None

        for _ in range(max_iter):
            next_vector = safe_normalize(residual.T @ (residual @ vector))
            if torch.norm(next_vector) <= EPSILON:
                break
            if previous_vector is not None:
                same_sign_delta = torch.norm(next_vector - previous_vector)
                flipped_sign_delta = torch.norm(next_vector + previous_vector)
                if torch.minimum(same_sign_delta, flipped_sign_delta) < tol:
                    vector = next_vector
                    break
            previous_vector = vector
            vector = next_vector

        residual_vector = residual @ vector
        sigma = torch.norm(residual_vector)
        if sigma <= EPSILON:
            break

        left_vector = residual_vector / sigma
        left_vectors.append(left_vector)
        singular_values.append(sigma)
        right_vectors.append(vector)
        residual = residual - sigma * torch.outer(left_vector, vector)

    if not singular_values:
        empty_u = torch.empty(out_features, 0, device=weight.device, dtype=original_dtype)
        empty_s = torch.empty(0, device=weight.device, dtype=original_dtype)
        empty_v = torch.empty(in_features, 0, device=weight.device, dtype=original_dtype)
        return empty_u, empty_s, empty_v

    left_matrix = torch.stack(left_vectors, dim=1).to(dtype=original_dtype)
    value_vector = torch.stack(singular_values).to(dtype=original_dtype)
    right_matrix = torch.stack(right_vectors, dim=1).to(dtype=original_dtype)
    return left_matrix, value_vector, right_matrix


def choose_truncated_rank(
    singular_values: torch.Tensor,
    default_rank: int,
    rank_ratio: Optional[float] = None,
    energy_threshold: Optional[float] = None,
) -> int:
    """
    Choose how many computed singular components to keep.
    Args:
        singular_values (torch.Tensor): Computed singular values in descending order.
        default_rank (int): Fixed fallback rank.
        rank_ratio (Optional[float]): Optional fraction of computed components to keep.
        energy_threshold (Optional[float]): Optional cumulative energy target in (0, 1].
    Returns:
        int: Number of singular components to keep.
    Algorithm:
        1. Return zero when no singular values are available.
        2. Prefer the smallest rank meeting the energy threshold when provided.
        3. Otherwise use rank ratio when provided, then the fixed default rank.
    """
    component_count = singular_values.numel()
    if component_count == 0:
        return 0

    if energy_threshold is not None:
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        energy = singular_values.float().pow(2)
        cumulative_energy = torch.cumsum(energy, dim=0) / torch.clamp(
            energy.sum(),
            min=EPSILON,
        )
        threshold = torch.tensor(energy_threshold, device=cumulative_energy.device)
        return int(torch.searchsorted(cumulative_energy, threshold).item() + 1)

    if rank_ratio is not None:
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError("rank_ratio must be in (0, 1]")
        return max(1, min(component_count, math.ceil(rank_ratio * component_count)))

    return max(1, min(default_rank, component_count))


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for a frozen linear layer."""

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        lora_alpha: int = 16,
        pretraining_mode: str = "standard",
        svd_rank_ratio: Optional[float] = None,
        svd_energy_threshold: Optional[float] = None,
        svd_max_iter: int = 100,
        svd_tol: float = 1e-6,
        svd_seed: Optional[int] = None,
    ) -> None:
        """
        Create a LoRA-augmented linear layer.
        Args:
            linear_layer (nn.Linear): Source linear layer to wrap.
            rank (int): LoRA adapter rank.
            lora_alpha (int): LoRA scaling numerator.
            pretraining_mode (str): Initialization mode: none, standard, or truncated_svd.
            svd_rank_ratio (Optional[float]): Optional ratio used by SVD rank truncation.
            svd_energy_threshold (Optional[float]): Optional SVD cumulative-energy threshold.
            svd_max_iter (int): Maximum power iterations per SVD component.
            svd_tol (float): Power-iteration convergence tolerance.
            svd_seed (Optional[int]): Optional seed for SVD initialization vectors.
        Returns:
            None: Initializes module state in place.
        Algorithm:
            1. Copy the frozen base weight and bias from the source layer.
            2. Allocate LoRA A and B trainable matrices.
            3. Initialize adapters using the selected pretraining mode.
        """
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if pretraining_mode not in VALID_PRETRAINING_MODES:
            raise ValueError(
                "pretraining_mode must be one of: 'none', 'standard', 'truncated_svd'"
            )

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scale = lora_alpha / rank
        self.pretraining_mode = pretraining_mode

        original_weight = linear_layer.weight.data.clone()
        self.weight = nn.Parameter(original_weight.clone(), requires_grad=False)
        self.bias = self._clone_frozen_bias(linear_layer)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))

        self.reset_lora_parameters(
            original_weight,
            svd_rank_ratio=svd_rank_ratio,
            svd_energy_threshold=svd_energy_threshold,
            svd_max_iter=svd_max_iter,
            svd_tol=svd_tol,
            svd_seed=svd_seed,
        )

    @staticmethod
    def _clone_frozen_bias(linear_layer: nn.Linear) -> Optional[nn.Parameter]:
        """
        Clone a source linear-layer bias as a frozen parameter.
        Args:
            linear_layer (nn.Linear): Source linear layer.
        Returns:
            Optional[nn.Parameter]: Frozen copied bias or None.
        Algorithm:
            1. Check whether the source layer has a bias.
            2. Return None when no bias exists.
            3. Clone the bias into a non-trainable parameter otherwise.
        """
        if linear_layer.bias is None:
            return None
        return nn.Parameter(linear_layer.bias.data.clone(), requires_grad=False)

    def reset_lora_parameters(
        self,
        original_weight: torch.Tensor,
        svd_rank_ratio: Optional[float],
        svd_energy_threshold: Optional[float],
        svd_max_iter: int,
        svd_tol: float,
        svd_seed: Optional[int],
    ) -> None:
        """
        Initialize LoRA parameters according to the configured mode.
        Args:
            original_weight (torch.Tensor): Copied source layer weight.
            svd_rank_ratio (Optional[float]): Optional SVD rank ratio.
            svd_energy_threshold (Optional[float]): Optional SVD energy threshold.
            svd_max_iter (int): Maximum SVD power iterations per component.
            svd_tol (float): SVD convergence tolerance.
            svd_seed (Optional[int]): Optional SVD random seed.
        Returns:
            None: Updates LoRA parameters in place.
        Algorithm:
            1. Zero both adapters for disabled LoRA pretraining.
            2. Use random A and zero B for standard LoRA.
            3. Use truncated SVD factors for structured initialization.
        """
        if self.pretraining_mode == "none":
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
            return

        if self.pretraining_mode == "standard":
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            nn.init.zeros_(self.lora_B)
            return

        self.init_with_truncated_svd(
            original_weight,
            rank_ratio=svd_rank_ratio,
            energy_threshold=svd_energy_threshold,
            max_iter=svd_max_iter,
            tol=svd_tol,
            seed=svd_seed,
        )

    def init_with_truncated_svd(
        self,
        weight: torch.Tensor,
        rank_ratio: Optional[float],
        energy_threshold: Optional[float],
        max_iter: int,
        tol: float,
        seed: Optional[int],
    ) -> None:
        """
        Initialize LoRA matrices from top truncated-SVD components.
        Args:
            weight (torch.Tensor): Source layer weight matrix W.
            rank_ratio (Optional[float]): Optional ratio of computed components to keep.
            energy_threshold (Optional[float]): Optional cumulative energy target.
            max_iter (int): Maximum power iterations per singular component.
            tol (float): Power-iteration convergence tolerance.
            seed (Optional[int]): Optional seed for right-vector initialization.
        Returns:
            None: Updates base residual weight and LoRA factors in place.
        Algorithm:
            1. Compute up to rank singular components from W.
            2. Choose the final kept rank by fixed, ratio, or energy logic.
            3. Factor W_k into balanced LoRA matrices B and A.
            4. Store W - W_k in the frozen path so the effective weight starts near W.
        """
        left_matrix, singular_values, right_matrix = truncated_svd_power_iteration(
            weight,
            rank=self.rank,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        kept_rank = choose_truncated_rank(
            singular_values,
            default_rank=self.rank,
            rank_ratio=rank_ratio,
            energy_threshold=energy_threshold,
        )

        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
        if kept_rank == 0:
            return

        left_matrix = left_matrix[:, :kept_rank]
        singular_values = singular_values[:kept_rank]
        right_matrix = right_matrix[:, :kept_rank]

        scaled_sqrt = torch.sqrt(torch.clamp(singular_values / self.scale, min=0.0))
        self.lora_B.data[:, :kept_rank] = (left_matrix * scaled_sqrt.unsqueeze(0)).to(
            dtype=self.lora_B.dtype
        )
        self.lora_A.data[:kept_rank, :] = (
            scaled_sqrt.unsqueeze(1) * right_matrix.T
        ).to(dtype=self.lora_A.dtype)

        low_rank_weight = self.scale * (
            self.lora_B.data[:, :kept_rank] @ self.lora_A.data[:kept_rank, :]
        )
        self.weight.data = (weight - low_rank_weight).to(dtype=self.weight.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the frozen base projection plus the trainable LoRA update.
        Args:
            inputs (torch.Tensor): Input tensor with trailing dimension equal to in_features.
        Returns:
            torch.Tensor: Output tensor after base and LoRA projections.
        Algorithm:
            1. Compute the frozen base linear projection.
            2. Compute the low-rank adapter projection A then B.
            3. Add scaled adapter output and optional bias.
        """
        base_output = inputs @ self.weight.T
        lora_output = (inputs @ self.lora_A.T) @ self.lora_B.T
        output = base_output + self.scale * lora_output
        if self.bias is not None:
            output = output + self.bias
        return output

    def merge(self) -> None:
        """
        Merge the LoRA update into the stored base weight.
        Args:
            None: This method uses module parameters only.
        Returns:
            None: Updates the frozen base weight in place.
        Algorithm:
            1. Compute scale * B @ A.
            2. Add the update to the base weight.
            3. Keep adapter parameters unchanged for caller-controlled lifecycle.
        """
        self.weight.data += (self.scale * (self.lora_B @ self.lora_A)).data


def get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Locate the parent module and child attribute name for a dotted module path.
    Args:
        model (nn.Module): Root model to traverse.
        module_name (str): Dotted child module path.
    Returns:
        Tuple[nn.Module, str]: Parent module and final child attribute name.
    Algorithm:
        1. Split the dotted module path into parts.
        2. Traverse all path parts except the final child name.
        3. Return the parent module and child name.
    """
    path_parts = module_name.split(".")
    parent_module = model
    for part in path_parts[:-1]:
        parent_module = getattr(parent_module, part)
    return parent_module, path_parts[-1]


def find_lora_target_modules(
    model: nn.Module,
    target_keywords: Sequence[str],
) -> list[str]:
    """
    Find linear module names that should be replaced by LoRA wrappers.
    Args:
        model (nn.Module): Model whose modules are scanned.
        target_keywords (Sequence[str]): Name fragments used to match target modules.
    Returns:
        list[str]: Dotted module names selected for replacement.
    Algorithm:
        1. Iterate over all named model modules.
        2. Keep linear layers whose names contain any target keyword.
        3. Return the collected module names.
    """
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(keyword in name for keyword in target_keywords)
    ]


def inject_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_keywords: Sequence[str] = ("query", "value"),
    pretraining_mode: str = "standard",
    svd_rank_ratio: Optional[float] = None,
    svd_energy_threshold: Optional[float] = None,
    svd_max_iter: int = 100,
    svd_tol: float = 1e-6,
    svd_seed: Optional[int] = None,
) -> nn.Module:
    """
    Replace selected linear layers with LoRA-wrapped linear layers.
    Args:
        model (nn.Module): Model to modify in place.
        r (int): LoRA rank.
        lora_alpha (int): LoRA scaling numerator.
        target_keywords (Sequence[str]): Module-name fragments to target.
        pretraining_mode (str): LoRA initialization mode.
        svd_rank_ratio (Optional[float]): Optional SVD rank ratio.
        svd_energy_threshold (Optional[float]): Optional SVD energy threshold.
        svd_max_iter (int): Maximum SVD power iterations per component.
        svd_tol (float): SVD convergence tolerance.
        svd_seed (Optional[int]): Optional SVD random seed.
    Returns:
        nn.Module: The same model with selected modules replaced.
    Algorithm:
        1. Find matching linear module names before mutation.
        2. Resolve each module parent and attribute name.
        3. Replace each target with a LoRALinear wrapper.
    """
    target_module_names = find_lora_target_modules(model, target_keywords)
    for module_name in target_module_names:
        parent_module, child_name = get_parent_module(model, module_name)
        source_layer = getattr(parent_module, child_name)
        setattr(
            parent_module,
            child_name,
            LoRALinear(
                source_layer,
                rank=r,
                lora_alpha=lora_alpha,
                pretraining_mode=pretraining_mode,
                svd_rank_ratio=svd_rank_ratio,
                svd_energy_threshold=svd_energy_threshold,
                svd_max_iter=svd_max_iter,
                svd_tol=svd_tol,
                svd_seed=svd_seed,
            ),
        )
    return model


def freeze_non_lora_params(model: nn.Module) -> None:
    """
    Freeze all model parameters except LoRA adapter matrices.
    Args:
        model (nn.Module): Model whose parameter flags are updated.
    Returns:
        None: Mutates requires_grad flags in place.
    Algorithm:
        1. Iterate through named parameters.
        2. Enable gradients for lora_A and lora_B parameters.
        3. Disable gradients for all remaining parameters.
    """
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_A" in name or "lora_B" in name


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge all LoRALinear adapter weights into their base weights.
    Args:
        model (nn.Module): Model containing optional LoRALinear modules.
    Returns:
        None: Updates LoRALinear base weights in place.
    Algorithm:
        1. Iterate through all model modules.
        2. Detect LoRALinear instances.
        3. Call merge on each detected adapter module.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """
    Count total and trainable parameters in a model.
    Args:
        model (nn.Module): Model whose parameters are counted.
    Returns:
        Dict[str, float]: Total count, trainable count, and trainable percentage.
    Algorithm:
        1. Sum the number of elements across all parameters.
        2. Sum elements only for parameters with gradients enabled.
        3. Compute the trainable percentage safely.
    """
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_percent = 100.0 * trainable_params / total_params if total_params else 0.0
    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percent": trainable_percent,
    }
