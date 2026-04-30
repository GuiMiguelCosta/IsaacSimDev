from __future__ import annotations

from pathlib import Path

import omni.ui as ui

from .constants import ROBOT_USD_PATH
from .path_utils import clean_repeated_absolute_path
from .policy_selection import available_policy_display_name, build_available_policy_configs
from .scene_objects import SCENE_OBJECT_OPTIONS, format_scene_object
from .ui_models import ComboBoxModel


class ExtensionUiMixin:
    """Omni UI model, widget, and control-refresh behavior for Extension."""

    def _initialize_ui_models(self, default_policies_folder: str) -> None:
        """Create Omni UI models and in-memory selection state.

        Input is the default policies folder; there is no output.
        This exists so startup wiring is separated from widget construction and
        event-handler behavior.
        """

        self._status_model = ui.SimpleStringModel("Press 'Load Scene' to create the scene.")
        self._queue_summary_model = ui.SimpleStringModel("Queue is empty.")
        default_policies_folder = clean_repeated_absolute_path(default_policies_folder)
        self._policies_folder_model = ui.SimpleStringModel(default_policies_folder)
        self._queue_config_path_model = ui.SimpleStringModel(
            self._queue_config_store.default_path(default_policies_folder)
        )
        self._available_policy_summary_model = ui.SimpleStringModel("Set a policies folder and load the scene.")
        self._available_policy_configs = []
        self._policy_selection_model = ComboBoxModel(["No available policies"])
        self._object_summary_model = ui.SimpleStringModel("Load the scene to edit table objects.")
        self._scene_object_keys = [object_key for object_key, _ in SCENE_OBJECT_OPTIONS]
        self._scene_object_labels = [display_name for _, display_name in SCENE_OBJECT_OPTIONS]
        self._add_object_combo_model = ComboBoxModel(self._scene_object_labels)
        self._remove_object_combo_model = ComboBoxModel(["Load scene first"])

    def _build_window(self) -> None:
        """Construct the extension window and bind callbacks.

        There are no inputs or outputs. This exists so the main Extension class
        does not have to own the verbose Omni UI layout tree.
        """

        self._window = ui.Window("IILAB Scene Extension", width=640, height=720)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Robot USD:")
                ui.Label(ROBOT_USD_PATH, word_wrap=True, style={"color": 0xFFBFBFBF, "font_size": 12})

                ui.Separator()

                ui.Label("Policies Folder:")
                with ui.HStack(spacing=8):
                    ui.StringField(self._policies_folder_model)
                    self._policy_scan_button = ui.Button("Scan", clicked_fn=self._on_scan_policies_clicked)

                ui.Separator()

                with ui.HStack(spacing=8):
                    self._load_button = ui.Button("Load Scene", clicked_fn=self._on_load_scene)
                    self._start_button = ui.Button("Start Simulation", clicked_fn=self._on_start_simulation)
                    self._policy_button = ui.Button("Run / Stop Policy", clicked_fn=self._on_policy_button_clicked)

                ui.Separator()

                ui.Label("Table Objects:")
                with ui.HStack(spacing=8):
                    with ui.VStack(spacing=4):
                        ui.Label("Add Object:")
                        self._add_object_combo = ui.ComboBox(self._add_object_combo_model)
                    self._object_add_button = ui.Button("Add Object", clicked_fn=self._on_add_scene_object_clicked)

                with ui.HStack(spacing=8):
                    with ui.VStack(spacing=4):
                        ui.Label("Remove Object:")
                        self._remove_object_combo = ui.ComboBox(self._remove_object_combo_model)
                    self._object_remove_button = ui.Button(
                        "Remove Object",
                        clicked_fn=self._on_remove_scene_object_clicked,
                    )

                ui.StringField(self._object_summary_model, multiline=True, height=64)

                ui.Separator()

                ui.Label("Available Policies:")
                with ui.HStack(spacing=8):
                    self._policy_selection_combo = ui.ComboBox(self._policy_selection_model)
                    self._queue_add_button = ui.Button("Add Selected", clicked_fn=self._on_add_policy_clicked)
                    self._queue_add_all_button = ui.Button("Add All", clicked_fn=self._on_add_all_policies_clicked)

                ui.StringField(self._available_policy_summary_model, multiline=True, height=64)

                with ui.HStack(spacing=8):
                    self._queue_remove_button = ui.Button("Remove Last", clicked_fn=self._on_remove_last_policy_clicked)
                    self._queue_clear_button = ui.Button("Clear Queue", clicked_fn=self._on_clear_policy_queue_clicked)

                ui.Label("Queued Policies:")
                ui.StringField(self._queue_summary_model, multiline=True, height=64)

                ui.Label("Queue YAML:")
                with ui.HStack(spacing=8):
                    ui.StringField(self._queue_config_path_model)
                    self._queue_export_button = ui.Button("Export Queue", clicked_fn=self._on_export_queue_clicked)
                    self._queue_import_button = ui.Button("Import Queue", clicked_fn=self._on_import_queue_clicked)

                ui.Separator()
                ui.Label("Status:")
                ui.StringField(self._status_model, multiline=True, height=84)

    def _set_status(self, text: str) -> None:
        """Set the user-facing status text.

        Input is the new status string; there is no output.
        This exists to keep status updates independent from the backing UI model.
        """

        self._status_model.set_value(text)

    def _refresh_buttons_state(self) -> None:
        """Enable or disable buttons based on scene, queue, and policy state.

        There are no inputs or outputs. This exists to prevent invalid actions
        such as editing objects while a policy controller is active.
        """

        self._load_button.enabled = not self._is_loading
        self._policy_scan_button.enabled = not self._is_loading
        self._start_button.enabled = not self._is_loading and self._scene is not None and not self._timeline.is_playing()
        self._policy_button.enabled = not self._is_loading and self._scene is not None
        scene_edit_enabled = not self._is_loading and self._scene is not None and self._policy_controller is None
        self._object_add_button.enabled = scene_edit_enabled
        self._object_remove_button.enabled = scene_edit_enabled and bool(self._scene.objects if self._scene else [])
        queue_edit_enabled = not self._is_loading and self._policy_controller is None
        self._queue_add_button.enabled = queue_edit_enabled and bool(self._available_policy_configs)
        self._queue_add_all_button.enabled = queue_edit_enabled and bool(self._available_policy_configs)
        self._queue_remove_button.enabled = queue_edit_enabled and bool(self._queued_policy_configs)
        self._queue_clear_button.enabled = queue_edit_enabled and bool(self._queued_policy_configs)
        self._queue_export_button.enabled = queue_edit_enabled
        self._queue_import_button.enabled = queue_edit_enabled

    def _policy_folder_path(self) -> Path | None:
        """Return the policies folder from the UI model.

        There are no inputs; the output is a Path or None.
        This exists to keep blank policies-folder handling consistent across scan
        and queue import.
        """

        raw_folder_path = str(self._policies_folder_model.as_string or "")
        folder_path = clean_repeated_absolute_path(raw_folder_path)
        if folder_path != raw_folder_path.strip():
            self._policies_folder_model.set_value(folder_path)
        if not folder_path:
            return None
        return Path(folder_path).expanduser()

    def _refresh_available_policy_controls(self, selected_object_prim_path: str | None = None) -> None:
        """Rebuild available-policy combo items and summary text.

        Input is an optional object prim path to prefer in the selection; there
        is no output. This exists after scene/object/policy-folder changes so the
        user only sees compatible policy-object pairs.
        """

        try:
            available_policy_configs, summary = build_available_policy_configs(self._scene, self._policy_folder_path())
        except Exception as exc:
            available_policy_configs = []
            summary = f"Could not scan policies: {exc}"

        selected_index = 0
        if selected_object_prim_path is not None:
            for index, policy_config in enumerate(available_policy_configs):
                if policy_config.target_object_prim_path == selected_object_prim_path:
                    selected_index = index
                    break

        self._available_policy_configs = available_policy_configs
        policy_items = (
            [available_policy_display_name(policy_config) for policy_config in available_policy_configs]
            if available_policy_configs
            else ["No available policies"]
        )
        self._policy_selection_model = ComboBoxModel(policy_items, selected_index)
        self._policy_selection_combo.model = self._policy_selection_model

        if available_policy_configs:
            summary_lines = [summary]
            summary_lines.extend(
                f"{index + 1}. {available_policy_display_name(policy_config)}"
                for index, policy_config in enumerate(available_policy_configs[:8])
            )
            if len(available_policy_configs) > 8:
                summary_lines.append(f"... and {len(available_policy_configs) - 8} more.")
            self._available_policy_summary_model.set_value("\n".join(summary_lines))
        else:
            self._available_policy_summary_model.set_value(summary)

    def _selected_add_object_key(self) -> str | None:
        """Return the object type selected in the add-object combo box.

        There are no inputs; the output is an object key or None.
        This exists to guard against stale UI selection indices.
        """

        selected_index = self._add_object_combo_model.selected_index
        if selected_index >= len(self._scene_object_keys):
            return None
        return self._scene_object_keys[selected_index]

    def _selected_remove_object_prim_path(self) -> str | None:
        """Return the prim path selected in the remove-object combo box.

        There are no inputs; the output is a prim path or None.
        This exists to handle empty scenes and stale selection indices safely.
        """

        if self._scene is None or not self._scene.objects:
            return None
        selected_index = self._remove_object_combo_model.selected_index
        if selected_index >= len(self._scene.objects):
            return None
        return self._scene.objects[selected_index].prim_path

    def _refresh_object_controls(self, selected_remove_prim_path: str | None = None) -> None:
        """Rebuild object removal choices and object summary text.

        Input is an optional prim path to select; there is no output.
        This exists after scene/object/import changes so the controls mirror the
        actual dynamic objects on the table.
        """

        if self._scene is None:
            remove_items = ["Load scene first"]
            selected_index = 0
            self._object_summary_model.set_value("Load the scene to edit table objects.")
        elif not self._scene.objects:
            remove_items = ["No objects in scene"]
            selected_index = 0
            self._object_summary_model.set_value("No table objects are currently loaded.")
        else:
            remove_items = [format_scene_object(scene_object) for scene_object in self._scene.objects]
            selected_index = 0
            if selected_remove_prim_path is not None:
                prim_paths = [scene_object.prim_path for scene_object in self._scene.objects]
                if selected_remove_prim_path in prim_paths:
                    selected_index = prim_paths.index(selected_remove_prim_path)
            summary_lines = [
                f"{index + 1}. {format_scene_object(scene_object)}"
                for index, scene_object in enumerate(self._scene.objects)
            ]
            self._object_summary_model.set_value("\n".join(summary_lines))

        self._remove_object_combo_model = ComboBoxModel(remove_items, selected_index)
        self._remove_object_combo.model = self._remove_object_combo_model
