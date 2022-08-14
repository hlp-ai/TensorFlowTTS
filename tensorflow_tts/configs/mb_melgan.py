"""Multi-band MelGAN Config object."""

from tensorflow_tts.configs import MelGANDiscriminatorConfig, MelGANGeneratorConfig


class MultiBandMelGANGeneratorConfig(MelGANGeneratorConfig):
    """Initialize Multi-band MelGAN Generator Config."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subbands = kwargs.pop("subbands", 4)
        self.taps = kwargs.pop("taps", 62)
        self.cutoff_ratio = kwargs.pop("cutoff_ratio", 0.142)
        self.beta = kwargs.pop("beta", 9.0)


class MultiBandMelGANDiscriminatorConfig(MelGANDiscriminatorConfig):
    """Initialize Multi-band MelGAN Discriminator Config."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
