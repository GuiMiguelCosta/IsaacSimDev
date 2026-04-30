from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .constants import CONTAINER_PRIM_PATH
from .path_utils import clean_repeated_absolute_path
from .policy_config import QueuedPolicyConfig
from .policy_discovery import (
    normalize_object_type,
    object_key_from_folder,
    object_key_from_policy_name,
    parse_policy_name,
)
from .scene_model import IILABScene
from .scene_objects import SCENE_OBJECT_SPECS

QUEUE_CONFIG_SCHEMA_VERSION = 1
DEFAULT_QUEUE_CONFIG_FILENAME = "iilab_policy_queue.yaml"


@dataclass(frozen=True)
class LoadedQueueConfig:
    """Validated queue YAML content ready to apply to the extension."""

    policies_folder: str
    queued_policy_configs: list[QueuedPolicyConfig]
    object_specs: list[tuple[str, str]]


class PolicyQueueConfigStore:
    """Serialize and deserialize policy queue YAML data."""

    def default_path(self, policies_folder: str) -> str:
        """Return the default YAML path for a policies folder.

        Input is the policies folder string; the output is a filesystem path.
        This exists to keep auto-save and export/import pointed at one location.
        """

        policies_folder = clean_repeated_absolute_path(policies_folder)
        folder_path = Path(policies_folder).expanduser() if policies_folder else Path.home()
        return str(folder_path / DEFAULT_QUEUE_CONFIG_FILENAME)

    def config_path(self, raw_path: str, policies_folder: str) -> Path:
        """Resolve the queue YAML path from UI text and policies folder.

        Inputs are the raw path and fallback policies folder; the output is a
        Path. This exists so blank UI input still has deterministic behavior.
        """

        raw_path = raw_path.strip()
        if not raw_path:
            raw_path = self.default_path(policies_folder)
        return Path(raw_path).expanduser()

    def build_data(
        self,
        queued_policy_configs: list[QueuedPolicyConfig],
        policies_folder: str,
        scene: IILABScene | None,
    ) -> dict:
        """Build the serializable queue YAML payload.

        Inputs are the queued configs, policies folder, and optional scene; the
        output is a dictionary for yaml.safe_dump. This exists to keep schema
        authorship separate from button callbacks.
        """

        return {
            "schema_version": QUEUE_CONFIG_SCHEMA_VERSION,
            "policies_folder": clean_repeated_absolute_path(policies_folder),
            "objects": self._object_entries(queued_policy_configs, scene),
            "queue": [
                {
                    "checkpoint_path": policy_config.checkpoint_path,
                    "policy_name": policy_config.policy_name,
                    "task_type": policy_config.task_type,
                    "robot_type": policy_config.robot_type,
                    "object_type": policy_config.object_type,
                    "target_object_prim_path": policy_config.target_object_prim_path,
                    "target_position_prim_path": policy_config.target_position_prim_path,
                    "norm_factor_min": policy_config.norm_factor_min,
                    "norm_factor_max": policy_config.norm_factor_max,
                }
                for policy_config in queued_policy_configs
            ],
        }

    def save(
        self,
        queue_config_path: Path,
        queued_policy_configs: list[QueuedPolicyConfig],
        policies_folder: str,
        scene: IILABScene | None,
    ) -> Path:
        """Write queue data to YAML.

        Inputs are the target path, queue, policies folder, and optional scene;
        the output is the written path. This exists to centralize file creation,
        parent-directory handling, and schema formatting.
        """

        queue_config_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                self.build_data(queued_policy_configs, policies_folder, scene),
                config_file,
                sort_keys=False,
            )
        return queue_config_path

    def load(self, queue_config_path: Path, fallback_policies_root: Path | None) -> LoadedQueueConfig:
        """Read and validate a queue YAML file.

        Inputs are the YAML path and fallback policies root; the output is a
        LoadedQueueConfig. This exists so import validation is testable without
        constructing the Omni extension window.
        """

        if not queue_config_path.exists():
            raise FileNotFoundError(f"Queue YAML not found: {queue_config_path}")

        with queue_config_path.open("r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file) or {}
        if not isinstance(config_data, dict):
            raise ValueError("Queue YAML root must be a mapping.")

        schema_version = int(config_data.get("schema_version", 1))
        if schema_version != QUEUE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported queue YAML schema_version {schema_version}; "
                f"expected {QUEUE_CONFIG_SCHEMA_VERSION}."
            )

        policies_folder = clean_repeated_absolute_path(str(config_data.get("policies_folder") or ""))
        policies_root = Path(policies_folder).expanduser() if policies_folder else fallback_policies_root
        object_types_by_prim_path = self._object_types_by_prim_path_from_config(config_data.get("objects", []))
        raw_queue = config_data.get("queue", config_data.get("policies", []))
        if raw_queue is None:
            raw_queue = []
        if not isinstance(raw_queue, list):
            raise ValueError("Queue YAML field 'queue' must be a list.")

        queued_policy_configs = [
            self._policy_config_from_queue_item(raw_policy_config, object_types_by_prim_path, policies_root, index)
            for index, raw_policy_config in enumerate(raw_queue)
        ]
        object_specs = self._object_specs_from_loaded_config(queued_policy_configs, object_types_by_prim_path)
        return LoadedQueueConfig(
            policies_folder=policies_folder,
            queued_policy_configs=queued_policy_configs,
            object_specs=object_specs,
        )

    def _object_entries(
        self,
        queued_policy_configs: list[QueuedPolicyConfig],
        scene: IILABScene | None,
    ) -> list[dict]:
        """Build unique object entries referenced by the queued policies.

        Inputs are queue configs and optional scene; the output is YAML-ready
        dictionaries. This exists so imported queues can recreate table objects
        before policy execution begins.
        """

        scene_objects_by_path = {scene_object.prim_path: scene_object for scene_object in scene.objects} if scene else {}
        object_entries = []
        seen_prim_paths = set()
        for policy_config in queued_policy_configs:
            prim_path = policy_config.target_object_prim_path
            if prim_path in seen_prim_paths:
                continue
            seen_prim_paths.add(prim_path)

            object_type = policy_config.object_type
            if not object_type and prim_path in scene_objects_by_path:
                object_type = scene_objects_by_path[prim_path].key
            display_name = SCENE_OBJECT_SPECS[object_type].display_name if object_type in SCENE_OBJECT_SPECS else object_type
            object_entries.append(
                {
                    "object_type": object_type,
                    "display_name": display_name,
                    "prim_path": prim_path,
                }
            )
        return object_entries

    def _object_types_by_prim_path_from_config(self, raw_objects) -> dict[str, str]:
        """Parse the YAML objects section into a prim-path map.

        Input is the raw objects field; the output maps prim paths to object
        keys. This exists to validate scene reconstruction data before mutating
        the current USD stage.
        """

        if raw_objects is None:
            return {}
        if not isinstance(raw_objects, list):
            raise ValueError("Queue YAML field 'objects' must be a list.")

        object_types_by_prim_path = {}
        for index, raw_object in enumerate(raw_objects):
            if not isinstance(raw_object, dict):
                raise ValueError(f"Queue YAML object entry {index + 1} must be a mapping.")
            prim_path = str(raw_object.get("prim_path") or "").strip()
            object_type = normalize_object_type(
                raw_object.get("object_type") or raw_object.get("key") or raw_object.get("type")
            )
            if not prim_path:
                raise ValueError(f"Queue YAML object entry {index + 1} is missing prim_path.")
            if object_type is None:
                raise ValueError(f"Queue YAML object entry {index + 1} has an unknown object_type.")
            object_types_by_prim_path[prim_path] = object_type
        return object_types_by_prim_path

    def _policy_config_from_queue_item(
        self,
        raw_policy_config,
        object_types_by_prim_path: dict[str, str],
        policies_root: Path | None,
        index: int,
    ) -> QueuedPolicyConfig:
        """Convert one YAML queue entry into a typed policy config.

        Inputs are the raw entry, object map, policies root, and entry index; the
        output is a QueuedPolicyConfig. This exists to keep object-type fallback
        heuristics and numeric parsing in one place.
        """

        if not isinstance(raw_policy_config, dict):
            raise ValueError(f"Queue YAML policy entry {index + 1} must be a mapping.")

        checkpoint_path = str(raw_policy_config.get("checkpoint_path") or "").strip()
        target_object_prim_path = str(raw_policy_config.get("target_object_prim_path") or "").strip()
        if not checkpoint_path:
            raise ValueError(f"Queue YAML policy entry {index + 1} is missing checkpoint_path.")
        if not target_object_prim_path:
            raise ValueError(f"Queue YAML policy entry {index + 1} is missing target_object_prim_path.")

        policy_name = str(raw_policy_config.get("policy_name") or Path(checkpoint_path).stem).strip()
        object_type = normalize_object_type(raw_policy_config.get("object_type"))
        if object_type is None:
            object_type = object_types_by_prim_path.get(target_object_prim_path)
        if object_type is None:
            object_type = object_key_from_policy_name(policy_name)
        if object_type is None and policies_root is not None:
            object_type = object_key_from_folder(Path(checkpoint_path), policies_root)
        if object_type is None:
            raise ValueError(
                f"Could not resolve object_type for queue policy entry {index + 1} "
                f"({target_object_prim_path})."
            )

        parsed_task_type, parsed_robot_type = parse_policy_name(policy_name, object_type)
        task_type = str(raw_policy_config.get("task_type") or parsed_task_type).strip()
        robot_type = str(raw_policy_config.get("robot_type") or parsed_robot_type).strip()
        target_position_prim_path = str(
            raw_policy_config.get("target_position_prim_path") or CONTAINER_PRIM_PATH
        ).strip()

        return QueuedPolicyConfig(
            checkpoint_path=checkpoint_path,
            target_object_prim_path=target_object_prim_path,
            target_position_prim_path=target_position_prim_path,
            policy_name=policy_name,
            task_type=task_type,
            robot_type=robot_type,
            object_type=object_type,
            norm_factor_min=self._parse_optional_config_float(raw_policy_config.get("norm_factor_min"), "norm_factor_min"),
            norm_factor_max=self._parse_optional_config_float(raw_policy_config.get("norm_factor_max"), "norm_factor_max"),
        )

    def _object_specs_from_loaded_config(
        self,
        queued_policy_configs: list[QueuedPolicyConfig],
        object_types_by_prim_path: dict[str, str],
    ) -> list[tuple[str, str]]:
        """Resolve table objects needed after queue import.

        Inputs are loaded policies and YAML object hints; the output is
        (object_type, prim_path) pairs. This exists so scene mutation happens
        once, after the full file has been validated.
        """

        object_specs: list[tuple[str, str]] = []
        seen_prim_paths = set()
        for policy_config in queued_policy_configs:
            if policy_config.target_object_prim_path in seen_prim_paths:
                continue
            seen_prim_paths.add(policy_config.target_object_prim_path)
            object_specs.append((policy_config.object_type, policy_config.target_object_prim_path))

        if not queued_policy_configs:
            object_specs = [(object_type, prim_path) for prim_path, object_type in object_types_by_prim_path.items()]
        return object_specs

    @staticmethod
    def _parse_optional_config_float(value, label: str) -> float | None:
        """Parse an optional numeric field from queue YAML.

        Inputs are the raw value and field label; the output is a float or None.
        This exists so imported configs report malformed normalization factors
        with the field name that needs fixing.
        """

        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a number when present.") from exc
