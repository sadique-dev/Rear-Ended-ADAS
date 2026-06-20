"""YAML configuration loader and typed settings dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# Project root is two levels above this file: src/utils/ -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class ModelConfig:
    """YOLO detection model settings."""

    name: str
    confidence: float
    iou: float
    classes: list[int]


@dataclass(frozen=True)
class TrackerConfig:
    """Multi-object tracker settings."""

    type: str
    persist: bool


@dataclass(frozen=True)
class VehicleWidthsConfig:
    """Real-world vehicle widths in metres by class name."""

    car: float
    motorcycle: float
    bus: float
    truck: float

    def get(self, class_name: str) -> float | None:
        """Return configured width for a class, or ``None`` if unknown."""
        return getattr(self, class_name, None)


@dataclass(frozen=True)
class CameraConfig:
    """Camera intrinsics and object dimension assumptions."""

    fov_horizontal_deg: float
    focal_length_px: float | None
    vehicle_widths_m: VehicleWidthsConfig
    assumed_vehicle_width_m: float


@dataclass(frozen=True)
class EstimationConfig:
    """Distance, speed, and TTC estimation parameters."""

    distance_ema_alpha: float
    distance_min_m: float
    distance_max_m: float
    speed_ema_alpha: float
    speed_min_mps: float
    speed_warmup_frames: int
    ttc_max_display_s: float


@dataclass(frozen=True)
class TTCConfig:
    """Time-To-Collision estimation settings."""

    minimum_speed_mps: float
    maximum_ttc_seconds: float
    display_infinity_as: str


@dataclass(frozen=True)
class RiskMessagesConfig:
    """Warning messages for each risk level."""

    safe: str
    caution: str
    danger: str


@dataclass(frozen=True)
class RiskConfig:
    """Collision risk classification settings."""

    danger_ttc_max_seconds: float
    caution_ttc_max_seconds: float
    messages: RiskMessagesConfig


@dataclass(frozen=True)
class VisualizationConfig:
    """HUD and overlay rendering settings."""

    show_all_tracks: bool
    hud_position: str
    danger_flash: bool


@dataclass(frozen=True)
class IOConfig:
    """Video output encoding settings."""

    output_codec: str
    output_fps: float | None


@dataclass(frozen=True)
class LeadVehicleROIConfig:
    """Normalized polygon defining the ego-lane region of interest."""

    points: list[tuple[float, float]]


@dataclass(frozen=True)
class LeadVehicleConfig:
    """Lead vehicle selection settings."""

    roi: LeadVehicleROIConfig


@dataclass(frozen=True)
class LoggingConfig:
    """Application logging settings."""

    level: str
    log_file: str | None


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object loaded from YAML."""

    model: ModelConfig
    tracker: TrackerConfig
    camera: CameraConfig
    estimation: EstimationConfig
    ttc: TTCConfig
    risk: RiskConfig
    visualization: VisualizationConfig
    io: IOConfig
    lead_vehicle: LeadVehicleConfig
    logging: LoggingConfig


def _require_section(data: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a required top-level config section or raise a clear error."""
    if section not in data:
        raise ValueError(f"Missing required config section: '{section}'")
    section_data = data[section]
    if not isinstance(section_data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return section_data


def _parse_risk_messages(data: dict[str, Any]) -> RiskMessagesConfig:
    """Parse risk warning messages from YAML."""
    messages_data = data.get("messages")
    if not isinstance(messages_data, dict):
        raise ValueError("risk.messages must be a mapping")

    for key in ("safe", "caution", "danger"):
        if key not in messages_data:
            raise ValueError(f"risk.messages.{key} is required")

    return RiskMessagesConfig(
        safe=str(messages_data["safe"]),
        caution=str(messages_data["caution"]),
        danger=str(messages_data["danger"]),
    )


def _validate_config(config: AppConfig) -> None:
    """Validate numeric ranges and enum-like string values."""
    if not 0.0 < config.model.confidence <= 1.0:
        raise ValueError("model.confidence must be in (0.0, 1.0]")
    if not 0.0 < config.model.iou <= 1.0:
        raise ValueError("model.iou must be in (0.0, 1.0]")
    if not config.model.classes:
        raise ValueError("model.classes must contain at least one class ID")

    if config.tracker.type not in {"bytetrack", "botsort"}:
        raise ValueError("tracker.type must be 'bytetrack' or 'botsort'")

    if not 0.0 < config.camera.fov_horizontal_deg < 180.0:
        raise ValueError("camera.fov_horizontal_deg must be in (0, 180)")
    if config.camera.assumed_vehicle_width_m <= 0.0:
        raise ValueError("camera.assumed_vehicle_width_m must be positive")

    if config.camera.focal_length_px is not None and config.camera.focal_length_px <= 0.0:
        raise ValueError("camera.focal_length_px must be positive when set")

    for class_name, width in (
        ("car", config.camera.vehicle_widths_m.car),
        ("motorcycle", config.camera.vehicle_widths_m.motorcycle),
        ("bus", config.camera.vehicle_widths_m.bus),
        ("truck", config.camera.vehicle_widths_m.truck),
    ):
        if width <= 0.0:
            raise ValueError(f"camera.vehicle_widths_m.{class_name} must be positive")

    if not 0.0 < config.estimation.distance_ema_alpha <= 1.0:
        raise ValueError("estimation.distance_ema_alpha must be in (0.0, 1.0]")
    if config.estimation.distance_min_m <= 0.0:
        raise ValueError("estimation.distance_min_m must be positive")
    if config.estimation.distance_max_m <= config.estimation.distance_min_m:
        raise ValueError("estimation.distance_max_m must exceed distance_min_m")

    if config.ttc.minimum_speed_mps <= 0.0:
        raise ValueError("ttc.minimum_speed_mps must be positive")
    if config.ttc.maximum_ttc_seconds <= 0.0:
        raise ValueError("ttc.maximum_ttc_seconds must be positive")
    if not config.ttc.display_infinity_as:
        raise ValueError("ttc.display_infinity_as must not be empty")
    if not 0.0 < config.estimation.speed_ema_alpha <= 1.0:
        raise ValueError("estimation.speed_ema_alpha must be in (0.0, 1.0]")
    if config.estimation.speed_warmup_frames < 1:
        raise ValueError("estimation.speed_warmup_frames must be >= 1")

    if config.risk.danger_ttc_max_seconds <= 0.0:
        raise ValueError("risk.danger_ttc_max_seconds must be positive")
    if config.risk.caution_ttc_max_seconds <= config.risk.danger_ttc_max_seconds:
        raise ValueError(
            "risk.caution_ttc_max_seconds must exceed danger_ttc_max_seconds"
        )
    for field_name, message in (
        ("safe", config.risk.messages.safe),
        ("caution", config.risk.messages.caution),
        ("danger", config.risk.messages.danger),
    ):
        if not message:
            raise ValueError(f"risk.messages.{field_name} must not be empty")

    if config.visualization.hud_position not in {
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    }:
        raise ValueError(
            "visualization.hud_position must be one of: "
            "top-left, top-right, bottom-left, bottom-right"
        )

    if config.io.output_fps is not None and config.io.output_fps <= 0.0:
        raise ValueError("io.output_fps must be positive when set")

    if len(config.lead_vehicle.roi.points) != 4:
        raise ValueError("lead_vehicle.roi.points must contain exactly 4 points")

    for x_norm, y_norm in config.lead_vehicle.roi.points:
        if not 0.0 <= x_norm <= 1.0 or not 0.0 <= y_norm <= 1.0:
            raise ValueError(
                "lead_vehicle.roi points must use normalized coordinates in [0, 1]"
            )


def _parse_lead_vehicle_roi(data: dict[str, Any]) -> LeadVehicleROIConfig:
    """Parse lead vehicle ROI polygon from YAML."""
    roi_data = data.get("roi")
    if not isinstance(roi_data, dict):
        raise ValueError("lead_vehicle.roi must be a mapping")

    raw_points = roi_data.get("points")
    if not isinstance(raw_points, list) or len(raw_points) != 4:
        raise ValueError("lead_vehicle.roi.points must be a list of 4 [x, y] pairs")

    points: list[tuple[float, float]] = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"lead_vehicle.roi.points[{index}] must be [x, y]"
            )
        points.append((float(point[0]), float(point[1])))

    return LeadVehicleROIConfig(points=points)


def _parse_vehicle_widths(data: dict[str, Any]) -> VehicleWidthsConfig:
    """Parse per-class vehicle width settings from YAML."""
    widths_data = data.get("vehicle_widths_m")
    if not isinstance(widths_data, dict):
        raise ValueError("camera.vehicle_widths_m must be a mapping")

    required = ("car", "motorcycle", "bus", "truck")
    for key in required:
        if key not in widths_data:
            raise ValueError(f"camera.vehicle_widths_m.{key} is required")

    return VehicleWidthsConfig(
        car=float(widths_data["car"]),
        motorcycle=float(widths_data["motorcycle"]),
        bus=float(widths_data["bus"]),
        truck=float(widths_data["truck"]),
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load and validate application configuration from a YAML file.

    Args:
        config_path: Path to a YAML config file. Defaults to
            ``config/default.yaml`` at the project root.

    Returns:
        Validated ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config structure or values are invalid.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    model_data = _require_section(raw, "model")
    tracker_data = _require_section(raw, "tracker")
    camera_data = _require_section(raw, "camera")
    estimation_data = _require_section(raw, "estimation")
    ttc_data = _require_section(raw, "ttc")
    risk_data = _require_section(raw, "risk")
    visualization_data = _require_section(raw, "visualization")
    io_data = _require_section(raw, "io")
    lead_vehicle_data = _require_section(raw, "lead_vehicle")
    logging_data = _require_section(raw, "logging")

    config = AppConfig(
        model=ModelConfig(
            name=str(model_data["name"]),
            confidence=float(model_data["confidence"]),
            iou=float(model_data["iou"]),
            classes=[int(c) for c in model_data["classes"]],
        ),
        tracker=TrackerConfig(
            type=str(tracker_data["type"]),
            persist=bool(tracker_data["persist"]),
        ),
        camera=CameraConfig(
            fov_horizontal_deg=float(camera_data["fov_horizontal_deg"]),
            focal_length_px=camera_data.get("focal_length_px"),
            vehicle_widths_m=_parse_vehicle_widths(camera_data),
            assumed_vehicle_width_m=float(camera_data["assumed_vehicle_width_m"]),
        ),
        estimation=EstimationConfig(
            distance_ema_alpha=float(estimation_data["distance_ema_alpha"]),
            distance_min_m=float(estimation_data["distance_min_m"]),
            distance_max_m=float(estimation_data["distance_max_m"]),
            speed_ema_alpha=float(estimation_data["speed_ema_alpha"]),
            speed_min_mps=float(estimation_data["speed_min_mps"]),
            speed_warmup_frames=int(estimation_data["speed_warmup_frames"]),
            ttc_max_display_s=float(estimation_data["ttc_max_display_s"]),
        ),
        ttc=TTCConfig(
            minimum_speed_mps=float(ttc_data["minimum_speed_mps"]),
            maximum_ttc_seconds=float(ttc_data["maximum_ttc_seconds"]),
            display_infinity_as=str(ttc_data["display_infinity_as"]),
        ),
        risk=RiskConfig(
            danger_ttc_max_seconds=float(risk_data["danger_ttc_max_seconds"]),
            caution_ttc_max_seconds=float(risk_data["caution_ttc_max_seconds"]),
            messages=_parse_risk_messages(risk_data),
        ),
        visualization=VisualizationConfig(
            show_all_tracks=bool(visualization_data["show_all_tracks"]),
            hud_position=str(visualization_data["hud_position"]),
            danger_flash=bool(visualization_data["danger_flash"]),
        ),
        io=IOConfig(
            output_codec=str(io_data["output_codec"]),
            output_fps=io_data["output_fps"],
        ),
        lead_vehicle=LeadVehicleConfig(
            roi=_parse_lead_vehicle_roi(lead_vehicle_data),
        ),
        logging=LoggingConfig(
            level=str(logging_data["level"]),
            log_file=logging_data["log_file"],
        ),
    )

    _validate_config(config)
    return config
