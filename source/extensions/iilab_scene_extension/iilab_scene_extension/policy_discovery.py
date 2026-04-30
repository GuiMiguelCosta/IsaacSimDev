from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from .scene_objects import SCENE_OBJECT_SPECS

CHECKPOINT_EXTENSIONS = {".pth", ".pt", ".ckpt"}


@dataclass(frozen=True)
class DiscoveredPolicy:
    """Metadata inferred from one robomimic checkpoint file."""

    checkpoint_path: str
    policy_name: str
    task_type: str
    robot_type: str
    object_key: str


def normalize_token(value: str) -> str:
    """Normalize free-form names for object and policy matching.

    Input is any string; the output is a lower-case alphanumeric token.
    This exists so filenames, folder names, display names, and prim prefixes can
    be compared without brittle punctuation assumptions.
    """

    return re.sub(r"[^a-z0-9]+", "", value.lower())


@lru_cache(maxsize=1)
def object_aliases() -> dict[str, str]:
    """Build lookup aliases for every known scene object.

    There are no inputs; the output maps normalized aliases to object keys.
    This exists to centralize object-type resolution and avoid rebuilding the
    alias table during large checkpoint scans.
    """

    aliases: dict[str, str] = {}
    for object_key, spec in SCENE_OBJECT_SPECS.items():
        for raw_alias in (object_key, spec.display_name, spec.prim_prefix):
            aliases[normalize_token(raw_alias)] = object_key
    return aliases


@lru_cache(maxsize=None)
def policy_name_suffixes(object_key: str) -> tuple[str, ...]:
    """Return suffixes that identify an object's policies by filename.

    Input is a scene object key; the output is a tuple of suffix strings.
    This exists to keep filename parsing resilient to display-name and prim-name
    variants used by training outputs.
    """

    spec = SCENE_OBJECT_SPECS[object_key]
    raw_suffixes = {
        object_key,
        object_key.replace("_", ""),
        spec.display_name.replace(" ", "_"),
        spec.display_name.replace(" ", ""),
        spec.prim_prefix,
    }
    return tuple(sorted({suffix.lower() for suffix in raw_suffixes if suffix}, key=len, reverse=True))


def object_key_from_folder(path: Path, policies_root: Path) -> str | None:
    """Infer an object key from the checkpoint's parent folders.

    Inputs are a checkpoint path and the policies root; the output is an object
    key or None. This exists as a fallback when policy filenames do not include
    the object name.
    """

    try:
        relative_parts = path.parent.relative_to(policies_root).parts
    except ValueError:
        relative_parts = path.parent.parts

    aliases = object_aliases()
    for part in reversed(relative_parts):
        object_key = aliases.get(normalize_token(part))
        if object_key is not None:
            return object_key
    return None


def object_key_from_policy_name(policy_name: str) -> str | None:
    """Infer an object key from a checkpoint stem.

    Input is the policy filename stem; the output is an object key or None.
    This exists so policy files can be matched to compatible table objects
    without requiring per-file YAML metadata.
    """

    normalized_policy_name = policy_name.lower()
    for object_key in SCENE_OBJECT_SPECS:
        for suffix in policy_name_suffixes(object_key):
            if normalized_policy_name == suffix or normalized_policy_name.endswith(f"_{suffix}"):
                return object_key
    return None


def normalize_object_type(raw_object_type) -> str | None:
    """Resolve a raw object type string to a known scene object key.

    Input may be None or any string-like value; the output is an object key or
    None. This exists to support both strict queue YAML and user-friendly aliases.
    """

    if raw_object_type is None:
        return None
    raw_object_type = str(raw_object_type).strip()
    if not raw_object_type:
        return None
    if raw_object_type in SCENE_OBJECT_SPECS:
        return raw_object_type
    return object_aliases().get(normalize_token(raw_object_type))


def parse_policy_name(policy_name: str, object_key: str) -> tuple[str, str]:
    """Split a checkpoint stem into task and robot labels.

    Inputs are the policy stem and resolved object key; the output is
    (task_type, robot_type). This exists to preserve labels such as
    task_axis/kuka while removing the object suffix used for matching.
    """

    prefix = policy_name
    normalized_policy_name = policy_name.lower()
    for suffix in policy_name_suffixes(object_key):
        if normalized_policy_name == suffix:
            prefix = ""
            break
        if normalized_policy_name.endswith(f"_{suffix}"):
            prefix = policy_name[: -(len(suffix) + 1)]
            break

    if not prefix:
        return "", ""

    parts = prefix.rsplit("_", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def discover_policies(policies_root: Path) -> list[DiscoveredPolicy]:
    """Find checkpoint files under a policies folder and classify them.

    Input is a policies root path; the output is a list of discovered policies.
    This exists to isolate filesystem scanning and naming heuristics from the UI.
    """

    policies: list[DiscoveredPolicy] = []
    for checkpoint_path in sorted(policies_root.rglob("*")):
        if not checkpoint_path.is_file() or checkpoint_path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
            continue

        policy_name = checkpoint_path.stem
        object_key = object_key_from_policy_name(policy_name) or object_key_from_folder(checkpoint_path, policies_root)
        if object_key is None:
            continue

        task_type, robot_type = parse_policy_name(policy_name, object_key)
        policies.append(
            DiscoveredPolicy(
                checkpoint_path=str(checkpoint_path),
                policy_name=policy_name,
                task_type=task_type,
                robot_type=robot_type,
                object_key=object_key,
            )
        )
    return policies
