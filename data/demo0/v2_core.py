from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class CandidateContext:
    grid_h: int
    grid_w: int
    similarity: Tensor
    topl_index: Tensor
    topl_similarity: Tensor
    candidate_ref_xy: Tensor
    displacement: Tensor
    appearance_weight: Tensor
    motion_vectors: Tensor
    slot_assignment: Tensor
    b4_features: Tensor
    b5_features: Tensor


@dataclass
class MethodPrediction:
    ref_xy: Tensor
    candidate_weight: Optional[Tensor]
    reject_prob: Tensor
    confidence: Tensor
    routed_feature: Tensor


class FrozenResNet18(nn.Module):
    """Frozen ImageNet feature encoder ending at stride-16 layer3."""

    output_channels = 256
    output_stride = 16

    def __init__(self, projection_dim: int, seed: int) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        self.body = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        for parameter in self.body.parameters():
            parameter.requires_grad_(False)
        self.body.eval()
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_matrix = torch.randn(self.output_channels, projection_dim, generator=generator)
        projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("projection", projection)

    @torch.inference_mode()
    def forward(self, rgb_0_1: Tensor) -> Tensor:
        normalized = (rgb_0_1 - self.mean) / self.std
        feature = self.body(normalized)
        feature = feature.permute(0, 2, 3, 1) @ self.projection
        feature = F.normalize(feature, dim=-1)
        return feature


class CandidateScorer(nn.Module):
    """Small equal-parameter scorer used by both B4 and B5."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(feature)
        return output[..., 0], output[..., 1]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def token_grid(height: int, width: int, device: torch.device) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def _initial_slots(displacement: Tensor, appearance: Tensor, count: int) -> Tensor:
    flat_d = displacement.reshape(-1, 2)
    flat_w = appearance.reshape(-1).clamp_min(1e-8)
    first = (flat_d * flat_w[:, None]).sum(0) / flat_w.sum()
    chosen = [first]
    for _ in range(1, count):
        centers = torch.stack(chosen, dim=0)
        distance = torch.cdist(flat_d, centers).square().amin(dim=1)
        index = torch.argmax(distance * flat_w)
        chosen.append(flat_d[index])
    return torch.stack(chosen, dim=0)


def soft_motion_slots(
    displacement: Tensor,
    appearance: Tensor,
    count: int,
    iterations: int,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Soft-EM over candidate 2-D translations.

    The non-differentiable farthest-point initialization is followed by fully
    differentiable soft assignments and weighted mean updates.
    """

    slots = _initial_slots(displacement.detach(), appearance.detach(), count)
    for _ in range(iterations):
        squared = (displacement[..., None, :] - slots[None, None, :, :]).square().sum(-1)
        assignment = torch.softmax(-squared / max(temperature, 1e-6), dim=-1)
        mass = appearance[..., None] * assignment
        denominator = mass.sum(dim=(0, 1)).clamp_min(1e-6)
        slots = (mass[..., None] * displacement[..., None, :]).sum(dim=(0, 1)) / denominator[:, None]
    squared = (displacement[..., None, :] - slots[None, None, :, :]).square().sum(-1)
    assignment = torch.softmax(-squared / max(temperature, 1e-6), dim=-1)
    return slots, assignment


def _neighbor_average(values: Tensor, grid_h: int, grid_w: int) -> Tensor:
    channels = values.shape[-1]
    image = values.reshape(grid_h, grid_w, channels).permute(2, 0, 1)[None]
    average = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    return average[0].permute(1, 2, 0).reshape(-1, channels)


def build_candidate_context(
    query: Tensor,
    reference: Tensor,
    grid_h: int,
    grid_w: int,
    top_l: int,
    slot_count: int,
    slot_iterations: int,
    appearance_temperature: float,
    slot_temperature: float,
    previous_slots: Optional[Tensor] = None,
) -> CandidateContext:
    query = F.normalize(query.float(), dim=-1)
    reference = F.normalize(reference.float(), dim=-1)
    similarity = query @ reference.T
    topl_similarity, topl_index = torch.topk(similarity, k=top_l, dim=-1)
    appearance_weight = torch.softmax(topl_similarity / appearance_temperature, dim=-1)

    coordinates = token_grid(grid_h, grid_w, query.device)
    candidate_ref_xy = coordinates[topl_index]
    displacement = coordinates[:, None, :] - candidate_ref_xy
    slots, assignment = soft_motion_slots(
        displacement,
        appearance_weight,
        slot_count,
        slot_iterations,
        slot_temperature,
    )

    squared = (displacement[..., None, :] - slots[None, None, :, :]).square().sum(-1)
    slot_fit = (assignment * torch.exp(-squared / (2.0 * slot_temperature))).sum(-1)
    token_slot = (appearance_weight[..., None] * assignment).sum(1)
    token_slot = token_slot / token_slot.sum(-1, keepdim=True).clamp_min(1e-6)
    neighborhood = _neighbor_average(token_slot, grid_h, grid_w)
    spatial_support = (assignment * neighborhood[:, None, :]).sum(-1)

    reverse_best = similarity.argmax(dim=0)
    query_ids = torch.arange(query.shape[0], device=query.device)[:, None]
    mutual = (reverse_best[topl_index] == query_ids).float()

    local_feature = _neighbor_average(query, grid_h, grid_w)
    local_coherence = F.cosine_similarity(query, local_feature, dim=-1).clamp(-1, 1)
    local_coherence = ((local_coherence + 1.0) * 0.5)[:, None].expand_as(topl_similarity)

    if previous_slots is None:
        temporal_fit = torch.ones_like(topl_similarity)
    else:
        temporal_distance = torch.cdist(displacement.reshape(-1, 2), previous_slots.float())
        temporal_fit = torch.exp(-temporal_distance.amin(-1) / 2.0).reshape_as(topl_similarity)

    margin = (topl_similarity[:, :1] - topl_similarity[:, 1:2]).expand_as(topl_similarity)
    entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / float(np.log(max(slot_count, 2)))

    width_norm = max(grid_w - 1, 1)
    height_norm = max(grid_h - 1, 1)
    dx = displacement[..., 0] / width_norm
    dy = displacement[..., 1] / height_norm
    magnitude = torch.sqrt(dx.square() + dy.square())
    query_xy = coordinates[:, None, :].expand_as(candidate_ref_xy)
    b4_features = torch.stack(
        (
            topl_similarity,
            appearance_weight,
            dx,
            dy,
            magnitude,
            query_xy[..., 0] / width_norm,
            query_xy[..., 1] / height_norm,
            margin,
        ),
        dim=-1,
    )
    b5_features = torch.stack(
        (
            topl_similarity,
            slot_fit,
            spatial_support,
            mutual,
            local_coherence,
            temporal_fit,
            margin,
            1.0 - entropy,
        ),
        dim=-1,
    )
    return CandidateContext(
        grid_h=grid_h,
        grid_w=grid_w,
        similarity=similarity,
        topl_index=topl_index,
        topl_similarity=topl_similarity,
        candidate_ref_xy=candidate_ref_xy,
        displacement=displacement,
        appearance_weight=appearance_weight,
        motion_vectors=slots,
        slot_assignment=assignment,
        b4_features=b4_features,
        b5_features=b5_features,
    )


def _candidate_prediction(
    context: CandidateContext,
    reference: Tensor,
    logits: Tensor,
    reject_logits: Optional[Tensor] = None,
) -> MethodPrediction:
    weight = torch.softmax(logits, dim=-1)
    ref_xy = (weight[..., None] * context.candidate_ref_xy).sum(1)
    gathered = reference[context.topl_index]
    routed = (weight[..., None] * gathered).sum(1)
    routed = F.normalize(routed, dim=-1)
    if reject_logits is None:
        reject_prob = torch.zeros(logits.shape[0], device=logits.device)
    else:
        reject_prob = torch.sigmoid((weight * reject_logits).sum(-1))
    confidence = (1.0 - reject_prob) * weight.amax(dim=-1)
    return MethodPrediction(ref_xy, weight, reject_prob, confidence, routed)


def predict_methods(
    context: CandidateContext,
    query: Tensor,
    reference: Tensor,
    b4_scorer: Optional[CandidateScorer],
    b5_scorer: Optional[CandidateScorer],
    appearance_temperature: float,
) -> Dict[str, MethodPrediction]:
    device = query.device
    count = query.shape[0]
    zero_reject = torch.zeros(count, device=device)
    coordinates = context.candidate_ref_xy
    gathered = reference[context.topl_index]

    b0_weight = F.one_hot(
        torch.zeros(count, dtype=torch.long, device=device),
        num_classes=context.topl_index.shape[1],
    ).float()
    b0 = MethodPrediction(
        ref_xy=coordinates[:, 0],
        candidate_weight=b0_weight,
        reject_prob=zero_reject,
        confidence=torch.sigmoid(context.topl_similarity[:, 0]),
        routed_feature=F.normalize(gathered[:, 0], dim=-1),
    )

    full_weight = torch.softmax(context.similarity / appearance_temperature, dim=-1)
    ref_coordinates = token_grid(context.grid_h, context.grid_w, device)
    b1_ref_xy = full_weight @ ref_coordinates
    b1_routed = F.normalize(full_weight @ reference, dim=-1)
    b1 = MethodPrediction(
        ref_xy=b1_ref_xy,
        candidate_weight=None,
        reject_prob=zero_reject,
        confidence=full_weight.amax(-1),
        routed_feature=b1_routed,
    )

    b2 = _candidate_prediction(
        context,
        reference,
        context.topl_similarity / appearance_temperature,
    )

    squared = (
        context.displacement[..., None, :] - context.motion_vectors[None, None, :, :]
    ).square().sum(-1)
    slot_fit = (context.slot_assignment * torch.exp(-squared / 3.0)).sum(-1)
    b3 = _candidate_prediction(
        context,
        reference,
        context.topl_similarity / appearance_temperature + 2.0 * slot_fit,
    )

    if b4_scorer is None:
        b4_logits = context.topl_similarity / appearance_temperature
        b4_reject_logits = None
    else:
        learned, b4_reject_logits = b4_scorer(context.b4_features)
        b4_logits = context.topl_similarity / appearance_temperature + learned
    b4 = _candidate_prediction(context, reference, b4_logits, b4_reject_logits)

    if b5_scorer is None:
        b5_logits = (
            context.topl_similarity / appearance_temperature
            + 2.0 * context.b5_features[..., 1]
            + 1.5 * context.b5_features[..., 2]
            + context.b5_features[..., 3]
            + 0.5 * context.b5_features[..., 5]
        )
        ambiguity = 1.0 - context.b5_features[..., 6].clamp(0.0, 1.0)
        b5_reject_logits = 2.0 * ambiguity - 2.0 * context.b5_features[..., 2]
    else:
        learned, b5_reject_logits = b5_scorer(context.b5_features)
        b5_logits = context.topl_similarity / appearance_temperature + learned
    b5 = _candidate_prediction(context, reference, b5_logits, b5_reject_logits)

    return {"B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5}
def append_reject_slot(slot_assignment: Tensor, reject_prob: Tensor) -> Tensor:
    reject = reject_prob[:, None, None].expand(slot_assignment.shape[0], slot_assignment.shape[1], 1)
    accepted = slot_assignment * (1.0 - reject_prob[:, None, None])
    return torch.cat((accepted, reject), dim=-1)


def stable_slot_permutation(current: Tensor, previous: Optional[Tensor]) -> np.ndarray:
    count = current.shape[0]
    if previous is None:
        order = np.lexsort((current[:, 1].detach().cpu().numpy(), current[:, 0].detach().cpu().numpy()))
        return order.astype(np.int64)
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(previous.float(), current.float()).detach().cpu().numpy()
    previous_id, current_id = linear_sum_assignment(cost)
    order = np.empty(count, dtype=np.int64)
    order[previous_id] = current_id
    return order
