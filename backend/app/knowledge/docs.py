"""Rule catalogue: per-rule defaults plus a short PyTorch documentation snippet for the fixer and UI."""

from app.schemas import DocRef

RULES: dict[str, dict] = {
    "EZ001": {
        "name": "shape-mismatch", "tier": "verified", "impact": "crash", "severity": "error",
        "description": "A layer receives a tensor whose size does not match what it was built for.",
        "docs": DocRef(
            title="torch.nn.Linear",
            url="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html",
            snippet="Linear(in_features, out_features) expects the last dimension of its input to equal in_features. "
                    "After Flatten, that is channels * height * width of the preceding feature map.",
        ),
    },
    "EZ002": {
        "name": "double-activation", "tier": "verified", "impact": "silent", "severity": "error",
        "description": "The model applies softmax or sigmoid and the loss applies it again internally.",
        "docs": DocRef(
            title="torch.nn.CrossEntropyLoss",
            url="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
            snippet="The input is expected to contain unnormalized logits for each class. CrossEntropyLoss applies "
                    "log_softmax internally, so applying softmax in the model flattens gradients and slows or stalls learning. "
                    "BCEWithLogitsLoss likewise applies sigmoid internally.",
        ),
    },
    "EZ003": {
        "name": "loss-goal-mismatch", "tier": "verified", "impact": "silent", "severity": "error",
        "description": "The loss function or output activation does not fit the stated task.",
        "docs": DocRef(
            title="Loss functions",
            url="https://pytorch.org/docs/stable/nn.html#loss-functions",
            snippet="Binary classification: one logit with BCEWithLogitsLoss. Multi-class: one logit per class with "
                    "CrossEntropyLoss. Multi-label: one logit per label with BCEWithLogitsLoss. Regression: no output "
                    "activation with MSELoss, L1Loss, or HuberLoss.",
        ),
    },
    "EZ004": {
        "name": "class-count-mismatch", "tier": "verified", "impact": "silent", "severity": "error",
        "description": "The number of classes differs between the model head and the rest of the code or goal.",
        "docs": DocRef(
            title="torch.nn.CrossEntropyLoss",
            url="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
            snippet="For C classes the model must output C channels. If the head outputs more channels than there are "
                    "label values, training runs without error but the extra outputs are never correct targets, and "
                    "metrics built for C classes will disagree with the model.",
        ),
    },
    "EZ005": {
        "name": "eval-augmentation-leak", "tier": "verified", "impact": "silent", "severity": "error",
        "description": "Random training augmentations are applied to validation or test data.",
        "docs": DocRef(
            title="torchvision.transforms",
            url="https://pytorch.org/vision/stable/transforms.html",
            snippet="Random transforms such as RandomHorizontalFlip, RandomCrop, and ColorJitter are for training. "
                    "Validation and test pipelines should be deterministic (for example Resize, CenterCrop, ToTensor, "
                    "Normalize) so metrics are stable and comparable between runs.",
        ),
    },
    "EZ006": {
        "name": "missing-zero-grad", "tier": "verified", "impact": "silent", "severity": "error",
        "description": "Gradients are never reset, so they accumulate across iterations.",
        "docs": DocRef(
            title="torch.optim.Optimizer.zero_grad",
            url="https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html",
            snippet="backward() adds to existing .grad values. Call optimizer.zero_grad() once per optimization step, "
                    "before backward(), unless you intend to accumulate gradients over several batches.",
        ),
    },
    "EZ007": {
        "name": "misplaced-zero-grad", "tier": "verified", "impact": "silent", "severity": "error",
        "description": "zero_grad() runs between backward() and step(), erasing the gradients before they are used.",
        "docs": DocRef(
            title="torch.optim.Optimizer.zero_grad",
            url="https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html",
            snippet="The usual order is zero_grad(), forward, loss, backward(), step(). Resetting between backward() "
                    "and step() means step() sees zero gradients and the model does not learn.",
        ),
    },
    "EZ008": {
        "name": "loss-accumulation-leak", "tier": "verified", "impact": "risk", "severity": "warning",
        "description": "A running total keeps every loss tensor and its graph alive, growing memory each step.",
        "docs": DocRef(
            title="Tensor.item",
            url="https://pytorch.org/docs/stable/generated/torch.Tensor.item.html",
            snippet="Adding a loss tensor to a Python total keeps its autograd graph referenced. Use loss.item() to "
                    "accumulate a plain number.",
        ),
    },
    "EZ009": {
        "name": "eval-mode-missing", "tier": "verified", "impact": "silent", "severity": "warning",
        "description": "Validation runs without torch.no_grad() or without model.eval().",
        "docs": DocRef(
            title="torch.no_grad and Module.eval",
            url="https://pytorch.org/docs/stable/generated/torch.no_grad.html",
            snippet="model.eval() switches Dropout and BatchNorm to inference behaviour. torch.no_grad() stops autograd "
                    "from recording, saving memory and time. Validation should use both.",
        ),
    },
    "EZ010": {
        "name": "ce-float-targets", "tier": "verified", "impact": "crash", "severity": "error",
        "description": "Class-index targets are cast to float before CrossEntropyLoss.",
        "docs": DocRef(
            title="torch.nn.CrossEntropyLoss",
            url="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
            snippet="Targets of shape (N) must be class indices of dtype long. Float targets are interpreted as class "
                    "probabilities and must then have the same shape as the input.",
        ),
    },
    "EZ011": {
        "name": "suspicious-hyperparameter", "tier": "advisory", "impact": "risk", "severity": "warning",
        "description": "A hyperparameter is outside the range that usually trains well.",
        "docs": DocRef(
            title="torch.optim",
            url="https://pytorch.org/docs/stable/optim.html",
            snippet="Adam and AdamW are typically used with learning rates between 1e-5 and 1e-3. BatchNorm needs more "
                    "than one sample per batch to compute statistics.",
        ),
    },
}


def rule_catalogue() -> list[dict]:
    return [
        {"id": rid, "name": r["name"], "description": r["description"], "tier": r["tier"],
         "impact": r["impact"], "docs_url": r["docs"].url}
        for rid, r in RULES.items()
    ]
