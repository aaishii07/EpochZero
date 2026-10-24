"""Rule catalogue: per-rule defaults plus a short PyTorch documentation snippet for the fixer and UI."""

from app.schemas import DocRef

RULES: dict[str, dict] = {
    "EZ000": {
        "name": "syntax-error", "tier": "verified", "impact": "crash", "severity": "error",
        "description": "The script is not valid Python, so it cannot start at all.",
        "docs": DocRef(
            title="Python syntax errors",
            url="https://docs.python.org/3/tutorial/errors.html#syntax-errors",
            snippet="Python reads the whole file before running it. A single syntax error stops the script before the "
                    "first line executes, so no other check can run until it is fixed.",
        ),
    },
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
    "EZ012": {
        "name": "python-runtime-error", "tier": "verified", "impact": "crash", "severity": "error",
        "description": "A name is used that is never defined, or a statement is invalid where it appears.",
        "docs": DocRef(
            title="NameError",
            url="https://docs.python.org/3/library/exceptions.html#NameError",
            snippet="Python raises NameError when a variable, function, or module is used before it is defined or "
                    "imported. The script stops at that line, often after minutes of data loading.",
        ),
    },
    "EZ013": {
        "name": "python-likely-mistake", "tier": "verified", "impact": "risk", "severity": "warning",
        "description": "A pattern that is legal Python but is almost always a mistake.",
        "docs": DocRef(
            title="pyflakes",
            url="https://pypi.org/project/pyflakes/",
            snippet="Redefining a name before using it, comparing with 'is' against a literal, or repeating a "
                    "dictionary key usually means a typo or a leftover line.",
        ),
    },
    "EZ101": {
        "name": "training-step-fails", "tier": "verified", "impact": "crash", "severity": "error",
        "description": "A real training step (forward, loss, backward) fails on a synthetic batch.",
        "docs": DocRef(
            title="Loss functions",
            url="https://pytorch.org/docs/stable/nn.html#loss-functions",
            snippet="The loss receives the model output and the targets from the dataset. Their shapes and dtypes "
                    "must match what the loss expects, for example (N, C, H, W) logits with (N, H, W) class indices "
                    "for CrossEntropyLoss in segmentation.",
        ),
    },
    "EZ102": {
        "name": "non-finite-loss", "tier": "probed", "impact": "silent", "severity": "error",
        "description": "The loss is NaN or infinite within the first training steps.",
        "docs": DocRef(
            title="Debugging NaN losses",
            url="https://pytorch.org/docs/stable/autograd.html#anomaly-detection",
            snippet="A NaN loss in the first steps usually comes from log of zero, division by zero, an activation "
                    "applied twice, or a loss given probabilities where it expects logits.",
        ),
    },
    "EZ103": {
        "name": "parameters-without-gradients", "tier": "probed", "impact": "silent", "severity": "warning",
        "description": "Some trainable layers receive no gradient, so they never learn.",
        "docs": DocRef(
            title="torch.Tensor.grad",
            url="https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html",
            snippet="A parameter keeps grad=None after backward() when its layer is not used in the forward pass or "
                    "its output is detached from the loss. The optimizer then never updates it.",
        ),
    },
    "EZ100": {
        "name": "code-review", "tier": "advisory", "impact": "risk", "severity": "warning",
        "description": "An issue found by the LLM code reviewer. No deterministic check proves it; review before applying.",
        "docs": None,
    },
}


# Pipeline area of each rule, used to group findings. Code-review findings choose their own area.
RULE_CATEGORY = {
    "EZ000": "syntax", "EZ001": "model", "EZ002": "loss", "EZ003": "loss", "EZ004": "model", "EZ005": "data",
    "EZ006": "training_loop", "EZ007": "training_loop", "EZ008": "training_loop", "EZ009": "evaluation",
    "EZ010": "loss", "EZ011": "optimization", "EZ012": "python", "EZ013": "python", "EZ100": "other",
    "EZ101": "loss", "EZ102": "loss", "EZ103": "model",
}


def rule_catalogue() -> list[dict]:
    return [
        {"id": rid, "name": r["name"], "description": r["description"], "tier": r["tier"],
         "impact": r["impact"], "docs_url": r["docs"].url if r["docs"] else None}
        for rid, r in RULES.items()
    ]
