from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment at each step during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Render the environment
        self.training_env.render('human')
        return True  # Returning True continues the training
