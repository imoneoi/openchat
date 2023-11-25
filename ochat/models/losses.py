import torch


# Accuracy
@torch.jit.script  # type: ignore
def weighted_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * (torch.argmax(logits, dim=-1) == labels)).sum()


# Loss function collections
@torch.jit.script  # type: ignore
def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * torch.nn.functional.cross_entropy(logits, labels, reduction="none")).sum()


@torch.jit.script  # type: ignore
def weighted_gold(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, beta: float = 0.05):
    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    p = torch.softmax(logits.detach(), -1)
    p = torch.gather(p, -1, labels.unsqueeze(-1)).squeeze(-1)
    p = torch.clamp(p, min=beta)

    return (weights * p * ce_loss).sum()


LOSS_FUNCTIONS = {
    "weighted_cross_entropy": weighted_cross_entropy,
    "weighted_gold": weighted_gold
}
