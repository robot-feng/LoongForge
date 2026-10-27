"""Console entry point for the Megatron-based LoongForge trainer."""


def main():
    """Parse the training configuration and run the selected trainer."""
    from loongforge.train import build_model_trainer, parse_train_args

    args = parse_train_args()
    build_model_trainer(args).train()
