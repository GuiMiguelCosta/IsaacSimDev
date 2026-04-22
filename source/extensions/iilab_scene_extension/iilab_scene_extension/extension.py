from __future__ import annotations

import asyncio
from pathlib import Path

import omni.ext
import omni.kit.app
import omni.physx
import omni.timeline
import omni.ui as ui

from .constants import (
    CONTAINER_PRIM_PATH,
    CUBE_3_PRIM_PATH,
    DEFAULT_ROBOMIMIC_CHECKPOINT,
    DEFAULT_ROBOMIMIC_NORM_FACTOR_MAX,
    DEFAULT_ROBOMIMIC_NORM_FACTOR_MIN,
    ROBOT_USD_PATH,
)
from .policy import RobomimicInferenceWorker, RobomimicPolicyController, guess_latest_checkpoint
from .scene import IILABScene, build_scene


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._timeline = omni.timeline.get_timeline_interface()
        self._physx_iface = omni.physx.get_physx_interface()
        self._scene: IILABScene | None = None
        self._policy_controller: RobomimicPolicyController | None = None
        self._is_loading = False
        self._physx_subscription = None
        self._timeline_subscription = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        self._status_model = ui.SimpleStringModel("Press 'Load Scene' to create the scene.")
        default_checkpoint = DEFAULT_ROBOMIMIC_CHECKPOINT or guess_latest_checkpoint()
        self._checkpoint_model = ui.SimpleStringModel(default_checkpoint)
        self._norm_min_model = ui.SimpleStringModel(DEFAULT_ROBOMIMIC_NORM_FACTOR_MIN)
        self._norm_max_model = ui.SimpleStringModel(DEFAULT_ROBOMIMIC_NORM_FACTOR_MAX)
        self._target_object_prim_model = ui.SimpleStringModel(CUBE_3_PRIM_PATH)
        self._target_position_prim_model = ui.SimpleStringModel(CONTAINER_PRIM_PATH)

        self._window = ui.Window("IILAB Scene Extension", width=520, height=430)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Robot USD:")
                ui.Label(ROBOT_USD_PATH, word_wrap=True, style={"color": 0xFFBFBFBF, "font_size": 12})

                ui.Separator()

                ui.Label("Policy Checkpoint:")
                ui.StringField(self._checkpoint_model)

                with ui.HStack(spacing=8):
                    with ui.VStack(spacing=4):
                        ui.Label("Target Object Prim:")
                        ui.StringField(self._target_object_prim_model)
                    with ui.VStack(spacing=4):
                        ui.Label("Target Position Prim:")
                        ui.StringField(self._target_position_prim_model)

                with ui.HStack(spacing=8):
                    with ui.VStack(spacing=4):
                        ui.Label("Norm Min:")
                        ui.StringField(self._norm_min_model)
                    with ui.VStack(spacing=4):
                        ui.Label("Norm Max:")
                        ui.StringField(self._norm_max_model)

                ui.Separator()

                with ui.HStack(spacing=8):
                    self._load_button = ui.Button("Load Scene", clicked_fn=self._on_load_scene)
                    self._start_button = ui.Button("Start Simulation", clicked_fn=self._on_start_simulation)
                    self._policy_button = ui.Button("Run / Stop Policy", clicked_fn=self._on_policy_button_clicked)

                ui.Separator()
                ui.Label("Status:")
                ui.StringField(self._status_model, multiline=True, height=96)

        self._refresh_buttons_state()

    def on_shutdown(self) -> None:
        self._stop_policy()
        self._scene = None
        self._timeline_subscription = None
        self._physx_subscription = None
        self._window = None

    def _set_status(self, text: str) -> None:
        self._status_model.set_value(text)

    def _refresh_buttons_state(self) -> None:
        self._load_button.enabled = not self._is_loading
        self._start_button.enabled = not self._is_loading and self._scene is not None and not self._timeline.is_playing()
        self._policy_button.enabled = not self._is_loading and self._scene is not None

    def _on_load_scene(self) -> None:
        if self._is_loading:
            return
        asyncio.ensure_future(self._on_load_scene_async())

    async def _on_load_scene_async(self) -> None:
        self._is_loading = True
        self._stop_policy()
        self._timeline.stop()
        self._set_status("Loading IILAB task scene...")
        self._refresh_buttons_state()

        await omni.kit.app.get_app().next_update_async()
        try:
            self._scene = build_scene()
            self._set_status("Scene loaded. Press 'Start Simulation' to run it.")
        except Exception as exc:
            self._timeline.stop()
            self._scene = None
            self._set_status(f"Failed to load scene: {exc}")
        finally:
            self._is_loading = False
            self._refresh_buttons_state()

    def _on_start_simulation(self) -> None:
        if self._scene is None:
            self._set_status("Load the scene first.")
            return

        if self._timeline.is_playing():
            self._set_status("Simulation is already running.")
        else:
            self._timeline.play()
            self._set_status("Simulation running.")

        self._refresh_buttons_state()

    def _on_policy_button_clicked(self) -> None:
        if self._policy_controller is not None:
            self._stop_policy("Robomimic policy stopped.")
            self._refresh_buttons_state()
            return

        if self._scene is None:
            self._set_status("Load the scene first.")
            return

        checkpoint_path = self._checkpoint_model.as_string.strip()
        if not checkpoint_path:
            self._set_status("Set a robomimic checkpoint path before starting the policy.")
            return

        target_object_prim_path = self._target_object_prim_model.as_string.strip()
        if not target_object_prim_path:
            self._set_status("Set the target object prim before starting the policy.")
            return

        target_position_prim_path = self._target_position_prim_model.as_string.strip()
        if not target_position_prim_path:
            self._set_status("Set the target position prim before starting the policy.")
            return

        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            self._set_status(f"Checkpoint not found: {checkpoint}")
            return

        worker = None
        try:
            norm_factor_min = self._parse_optional_float(self._norm_min_model.as_string.strip(), "Norm Min")
            norm_factor_max = self._parse_optional_float(self._norm_max_model.as_string.strip(), "Norm Max")
            if (norm_factor_min is None) != (norm_factor_max is None):
                raise ValueError("Set both normalization values or leave both empty.")

            worker = RobomimicInferenceWorker(
                checkpoint_path=str(checkpoint),
                norm_factor_min=norm_factor_min,
                norm_factor_max=norm_factor_max,
            )
            self._policy_controller = RobomimicPolicyController(
                scene=self._scene,
                worker=worker,
                target_object_prim_path=target_object_prim_path,
                target_position_prim_path=target_position_prim_path,
                status_callback=self._set_status,
            )
        except Exception as exc:
            if worker is not None:
                worker.close()
            self._stop_policy()
            self._set_status(f"Failed to start robomimic policy: {exc}")
            self._refresh_buttons_state()
            return

        if not self._timeline.is_playing():
            self._timeline.play()

        observation_keys = ", ".join(self._policy_controller.policy_observation_keys)
        self._set_status(
            "Robomimic policy running.\n"
            f"Target object: {target_object_prim_path}\n"
            f"Target position: {target_position_prim_path}\n"
            f"Policy inputs: {observation_keys}"
        )
        self._refresh_buttons_state()

    def _on_timeline_event(self, event) -> None:
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if self._physx_subscription is None:
                self._physx_subscription = self._physx_iface.subscribe_physics_step_events(self._on_physics_step)
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._physx_subscription = None
            if self._policy_controller is not None:
                try:
                    self._policy_controller.reset()
                except Exception:
                    self._stop_policy("Robomimic policy stopped after a reset.")
        self._refresh_buttons_state()

    def _on_physics_step(self, step: float) -> None:
        if self._policy_controller is None:
            return

        try:
            self._policy_controller.on_physics_step(step)
            if self._policy_controller.is_task_complete():
                completion_status = self._policy_controller.completion_status_message or "Policy is complete."
                self._stop_policy(completion_status)
                self._refresh_buttons_state()
        except Exception as exc:
            self._stop_policy(f"Robomimic policy error: {exc}")
            self._refresh_buttons_state()

    def _stop_policy(self, status_message: str | None = None) -> None:
        if self._policy_controller is not None:
            try:
                self._policy_controller.hold_current_pose()
            except Exception:
                pass
            try:
                self._policy_controller.close()
            except Exception:
                pass
            self._policy_controller = None
        if status_message is not None:
            self._set_status(status_message)

    @staticmethod
    def _parse_optional_float(value: str, label: str) -> float | None:
        if value == "":
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
