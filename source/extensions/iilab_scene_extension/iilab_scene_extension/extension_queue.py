from __future__ import annotations

import asyncio
from pathlib import Path

import omni.kit.app

from .policy_config import QueuedPolicyConfig
from .scene import build_scene, set_scene_objects


class ExtensionQueueMixin:
    """Queue mutation and YAML import/export behavior for Extension."""

    def _on_export_queue_clicked(self) -> None:
        """Handle the Export Queue button.

        There are no inputs or outputs. This exists to translate file-write
        errors into status text while keeping the UI responsive.
        """

        try:
            queue_config_path = self._save_queue_config()
        except Exception as exc:
            self._set_status(f"Failed to export queue YAML: {exc}")
            self._refresh_buttons_state()
            return

        self._set_status(
            f"Exported {len(self._queued_policy_configs)} queued "
            f"polic{'y' if len(self._queued_policy_configs) == 1 else 'ies'}.\n"
            f"Queue YAML: {queue_config_path}"
        )
        self._refresh_buttons_state()

    def _on_import_queue_clicked(self) -> None:
        """Schedule asynchronous queue import from the Import Queue button.

        There are no inputs or outputs. This exists so YAML reading and scene
        mutation happen outside the immediate UI callback.
        """

        if self._is_loading:
            return
        asyncio.ensure_future(self._on_import_queue_async())

    async def _on_import_queue_async(self) -> None:
        """Import queue YAML, rebuild scene objects, and refresh controls.

        There are no inputs or outputs. This exists to give Kit an update before
        heavier file and stage work, reducing UI stalls.
        """

        self._is_loading = True
        self._stop_policy_execution()
        self._timeline.stop()
        self._set_status("Importing queue YAML...")
        self._refresh_buttons_state()

        await omni.kit.app.get_app().next_update_async()
        try:
            queue_config_path = self._queue_config_path()
            queue_count, object_count = self._load_queue_config(queue_config_path)
            self._update_queue_summary()
            self._set_status(
                f"Imported {queue_count} queued polic{'y' if queue_count == 1 else 'ies'} "
                f"from {queue_config_path}.\n"
                f"Loaded {object_count} table object{'s' if object_count != 1 else ''} from the queue config."
            )
        except Exception as exc:
            self._set_status(f"Failed to import queue YAML: {exc}")
        finally:
            self._is_loading = False
            self._refresh_object_controls()
            self._refresh_available_policy_controls()
            self._refresh_buttons_state()

    def _on_add_policy_clicked(self) -> None:
        """Add the selected compatible policy/object pair to the queue.

        There are no inputs or outputs. This exists to deduplicate queue entries
        and persist the queue after UI mutation.
        """

        try:
            policy_config = self._build_policy_config_from_models()
        except Exception as exc:
            self._set_status(f"Could not add policy to queue: {exc}")
            self._refresh_buttons_state()
            return

        policy_key = (policy_config.checkpoint_path, policy_config.target_object_prim_path)
        existing_keys = {
            (queued_policy.checkpoint_path, queued_policy.target_object_prim_path)
            for queued_policy in self._queued_policy_configs
        }
        if policy_key in existing_keys:
            self._set_status(f"Policy is already queued.\n{policy_config.display_name}")
            self._refresh_buttons_state()
            return

        self._queued_policy_configs.append(policy_config)
        self._update_queue_summary()
        queue_save_status = self._save_queue_config_for_status()
        self._set_status(
            f"Added policy {len(self._queued_policy_configs)} to the queue.\n"
            f"Checkpoint: {policy_config.checkpoint_path}\n"
            f"Target object: {policy_config.target_object_prim_path}\n"
            f"Target position: {policy_config.target_position_prim_path}\n"
            f"{queue_save_status}"
        )
        self._refresh_buttons_state()

    def _on_add_all_policies_clicked(self) -> None:
        """Add every currently compatible policy/object pair to the queue.

        There are no inputs or outputs. This exists for batch workflows while
        still preserving deduplication against existing queue entries.
        """

        if not self._available_policy_configs:
            self._set_status("No available policies to add. Load the scene and scan a policies folder first.")
            self._refresh_buttons_state()
            return

        existing_keys = {
            (policy_config.checkpoint_path, policy_config.target_object_prim_path)
            for policy_config in self._queued_policy_configs
        }
        added_count = 0
        for policy_config in self._available_policy_configs:
            policy_key = (policy_config.checkpoint_path, policy_config.target_object_prim_path)
            if policy_key in existing_keys:
                continue
            self._queued_policy_configs.append(policy_config)
            existing_keys.add(policy_key)
            added_count += 1

        self._update_queue_summary()
        queue_save_status = self._save_queue_config_for_status()
        self._set_status(
            f"Added {added_count} available polic{'y' if added_count == 1 else 'ies'} to the queue.\n"
            f"{queue_save_status}"
        )
        self._refresh_buttons_state()

    def _on_remove_last_policy_clicked(self) -> None:
        """Remove the last queued policy.

        There are no inputs or outputs. This exists as a simple correction path
        that preserves ordering semantics of the policy chain.
        """

        if not self._queued_policy_configs:
            self._set_status("Policy queue is already empty.")
            self._refresh_buttons_state()
            return

        removed_policy = self._queued_policy_configs.pop()
        self._update_queue_summary()
        queue_save_status = self._save_queue_config_for_status()
        self._set_status(f"Removed the last queued policy.\n{removed_policy.display_name}\n{queue_save_status}")
        self._refresh_buttons_state()

    def _on_clear_policy_queue_clicked(self) -> None:
        """Clear every queued policy.

        There are no inputs or outputs. This exists to reset chain setup without
        rebuilding the scene.
        """

        if not self._queued_policy_configs:
            self._set_status("Policy queue is already empty.")
            self._refresh_buttons_state()
            return

        removed_count = len(self._queued_policy_configs)
        self._queued_policy_configs.clear()
        self._update_queue_summary()
        queue_save_status = self._save_queue_config_for_status()
        self._set_status(
            f"Cleared {removed_count} queued polic{'y' if removed_count == 1 else 'ies'}.\n{queue_save_status}"
        )
        self._refresh_buttons_state()

    def _build_policy_config_from_models(self) -> QueuedPolicyConfig:
        """Return the selected available policy config.

        There are no inputs; the output is a QueuedPolicyConfig.
        This exists to convert combo-box selection state into a typed policy
        config while refreshing stale scan results when needed.
        """

        if not self._available_policy_configs:
            self._refresh_available_policy_controls()

        if not self._available_policy_configs:
            raise ValueError("No compatible policies found. Load the scene and scan a policies folder first.")

        selected_index = self._policy_selection_model.selected_index
        if selected_index >= len(self._available_policy_configs):
            raise ValueError("Select a compatible policy before adding it to the queue.")

        return self._available_policy_configs[selected_index]

    def _queue_config_path(self) -> Path:
        """Resolve the queue YAML path from the UI model.

        There are no inputs; the output is a Path.
        This exists so blank path fields are replaced by a deterministic default
        and written back to the UI.
        """

        raw_path = self._queue_config_path_model.as_string.strip()
        if not raw_path:
            raw_path = self._queue_config_store.default_path(self._policies_folder_model.as_string)
            self._queue_config_path_model.set_value(raw_path)
        return self._queue_config_store.config_path(raw_path, self._policies_folder_model.as_string)

    def _save_queue_config(self) -> Path:
        """Persist the current queue to YAML.

        There are no inputs; the output is the written path.
        This exists as the shared save path for explicit export and automatic
        queue updates.
        """

        return self._queue_config_store.save(
            self._queue_config_path(),
            self._queued_policy_configs,
            self._policies_folder_model.as_string,
            self._scene,
        )

    def _save_queue_config_for_status(self) -> str:
        """Save the queue and convert success/failure into a status line.

        There are no inputs; the output is a status string.
        This exists so queue-mutating callbacks can report persistence without
        duplicating exception handling.
        """

        try:
            return f"Queue YAML: {self._save_queue_config()}"
        except Exception as exc:
            return f"Queue YAML save failed: {exc}"

    def _load_queue_config(self, queue_config_path: Path) -> tuple[int, int]:
        """Load queue YAML and apply its object layout to the scene.

        Input is the YAML path; the output is (queue_count, object_count).
        This exists to validate queue data before replacing scene objects and
        queued policy state.
        """

        loaded_config = self._queue_config_store.load(queue_config_path, self._policy_folder_path())
        if loaded_config.policies_folder:
            self._policies_folder_model.set_value(loaded_config.policies_folder)

        if self._scene is None:
            self._scene = build_scene()
        set_scene_objects(self._scene, loaded_config.object_specs)
        self._queued_policy_configs = loaded_config.queued_policy_configs
        return len(loaded_config.queued_policy_configs), len(loaded_config.object_specs)

    def _update_queue_summary(self) -> None:
        """Refresh the queued policy summary text.

        There are no inputs or outputs. This exists so queue state and visible
        status remain synchronized after every mutation.
        """

        if not self._queued_policy_configs:
            self._queue_summary_model.set_value("Queue is empty.")
            return

        queue_lines = [
            f"{index + 1}. {policy_config.display_name}"
            for index, policy_config in enumerate(self._queued_policy_configs)
        ]
        self._queue_summary_model.set_value("\n".join(queue_lines))

    def _prune_queue_for_scene_objects(self) -> None:
        """Drop queued policies whose target objects no longer exist.

        There are no inputs or outputs. This exists after object removal so the
        queue cannot later reference deleted prims.
        """

        if self._scene is None:
            self._queued_policy_configs.clear()
            return

        valid_prim_paths = {scene_object.prim_path for scene_object in self._scene.objects}
        self._queued_policy_configs = [
            policy_config
            for policy_config in self._queued_policy_configs
            if policy_config.target_object_prim_path in valid_prim_paths
        ]
