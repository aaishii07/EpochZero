"""What a correct setup looks like for each task. Plain data, no LLM."""

EXPECTATIONS: dict[str, dict] = {
    "binary_classification": {
        "losses": {"BCEWithLogitsLoss", "CrossEntropyLoss", "BCELoss"},
        "preferred_loss": "BCEWithLogitsLoss",
        "bad_activations": {"softmax", "log_softmax", "relu"},
        "head_out": 1,
    },
    "multiclass_classification": {
        "losses": {"CrossEntropyLoss", "NLLLoss"},
        "preferred_loss": "CrossEntropyLoss",
        "bad_activations": {"sigmoid", "relu"},
        "head_out": "num_classes",
    },
    "multilabel_classification": {
        "losses": {"BCEWithLogitsLoss", "BCELoss"},
        "preferred_loss": "BCEWithLogitsLoss",
        "bad_activations": {"softmax", "log_softmax", "relu"},
        "head_out": "num_classes",
    },
    "regression": {
        "losses": {"MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss"},
        "preferred_loss": "MSELoss",
        "bad_activations": {"softmax", "log_softmax", "sigmoid"},
        "head_out": None,
    },
    "segmentation": {
        "losses": {"CrossEntropyLoss", "BCEWithLogitsLoss", "DiceLoss", "NLLLoss", "FocalLoss"},
        "preferred_loss": "CrossEntropyLoss",
        "bad_activations": {"relu"},
        "head_out": "num_classes",
    },
    "image_to_image": {
        "losses": {"L1Loss", "MSELoss", "SmoothL1Loss", "HuberLoss", "CharbonnierLoss", "PSNRLoss", "SSIMLoss",
                   "PerceptualLoss", "L1CharbonnierLoss"},
        "preferred_loss": "L1Loss",
        "bad_activations": {"softmax", "log_softmax"},
        "head_out": None,
    },
}
