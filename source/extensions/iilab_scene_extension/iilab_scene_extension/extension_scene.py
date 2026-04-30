from __future__ import annotations

import asyncio

import omni.kit.app

from .scene import add_scene_object, build_scene, format_scene_object, remove_scene_object


class ExtensionSceneMixin:
    """Scene lifecycle and table-object edit behavior for Extension."""

    def _on_load_scene(self) -> None:
        """Schedule a full scene load from the Load Scene button.

        There are no inputs or outputs. This exists to keep the UI callback fast
        while the async task performs stage work.
        """

        if self._is_loading:
            return
        asyncio.ensure_future(self._on_load_scene_async())

    async def _on_load_scene_async(self) -> None:
        """Build a fresh scene and reset dependent queue/UI state.

        There are no inputs or outputs. This exists to create a clean stage while
        safely stopping simulation and any active policy first.
        """

        self._is_loading = True
        self._stop_policy_execution()
        self._timeline.stop()
        self._set_status("Loading IILAB task scene...")
        self._refresh_buttons_state()

        await omni.kit.app.get_app().next_update_async()
        try:
            self._scene = build_scene()
            self._queued_policy_configs.clear()
            self._update_queue_summary()
            self._set_status("Scene loaded. Press 'Start Simulation' to run it.")
        except Exception as exc:
            self._timeline.stop()
            self._scene = None
            self._set_status(f"Failed to load scene: {exc}")
        finally:
            self._is_loading = False
            self._refresh_object_controls()
            self._refresh_available_policy_controls()
            self._refresh_buttons_state()

    def _on_add_scene_object_clicked(self) -> None:
        """Add the selected table object to the loaded scene.

        There are no inputs or outputs. This exists to stop incompatible running
        state, mutate the scene, refresh policy matches, and report errors.
        """

        if self._scene is None:
            self._set_status("Load the scene first.")
            self._refresh_buttons_state()
            return

        object_key = self._selected_add_object_key()
        if object_key is None:
            self._set_status("Select an object to add.")
            self._refresh_buttons_state()
            return

        self._stop_policy_execution()
        self._timeline.stop()
        try:
            scene_object = add_scene_object(self._scene, object_key)
        except Exception as exc:
            self._set_status(f"Failed to add object: {exc}")
            self._refresh_buttons_state()
            return

        self._refresh_object_controls(selected_remove_prim_path=scene_object.prim_path)
        self._refresh_available_policy_controls(selected_object_prim_path=scene_object.prim_path)
        self._set_status(f"Added {format_scene_object(scene_object)}.\nScene reset and object positions randomized.")
        self._refresh_buttons_state()

    def _on_remove_scene_object_clicked(self) -> None:
        """Remove the selected table object from the loaded scene.

        There are no inputs or outputs. This exists to keep the policy queue and
        compatible-policy list aligned with objects still present in the stage.
        """

        if self._scene is None:
            self._set_status("Load the scene first.")
            self._refresh_buttons_state()
            return

        prim_path = self._selected_remove_object_prim_path()
        if prim_path is None:
            self._set_status("No scene objects are available to remove.")
            self._refresh_buttons_state()
            return

        self._stop_policy_execution()
        self._timeline.stop()
        try:
            removed_object = remove_scene_object(self._scene, prim_path)
        except Exception as exc:
            self._set_status(f"Failed to remove object: {exc}")
            self._refresh_buttons_state()
            return

        self._refresh_object_controls()
        self._prune_queue_for_scene_objects()
        self._refresh_available_policy_controls()
        self._update_queue_summary()
        queue_save_status = self._save_queue_config_for_status()
        self._set_status(
            f"Removed {format_scene_object(removed_object)}.\n"
            f"Scene reset and object positions randomized.\n"
            f"{queue_save_status}"
        )
        self._refresh_buttons_state()

    def _on_start_simulation(self) -> None:
        """Start timeline playback when a scene is loaded.

        There are no inputs or outputs. This exists to guard the Start
        Simulation button and provide clear status if playback is already active.
        """

        if self._scene is None:
            self._set_status("Load the scene first.")
            return

        if self._timeline.is_playing():
            self._set_status("Simulation is already running.")
        else:
            self._timeline.play()
            self._set_status("Simulation running.")

        self._refresh_buttons_state()

    def _on_scan_policies_clicked(self) -> None:
        """Refresh compatible policies from the current policies folder.

        There are no inputs or outputs. This exists as the manual rescan path
        after the user edits the policies folder text field.
        """

        self._refresh_available_policy_controls()
        self._set_status(self._available_policy_summary_model.as_string)
        self._refresh_buttons_state()
