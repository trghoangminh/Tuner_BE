from dataclasses import dataclass


@dataclass
class Settings:
    a4_hz: float = 440.0
    sample_rate: int = 44100
    hop_length: int = 512
    frame_length: int = 2048
    algorithm: str = "acf"  # options: "acf", "yin"
    smoothing: str = "none"  # options: "none", "ema", "median"
    ema_alpha: float = 0.3
    median_window: int = 5


settings = Settings()


