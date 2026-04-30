from __future__ import annotations

import omni.timeline

from .policy import RobomimicInferenceWorker, RobomimicPolicyController
from .policy_config import QueuedPolicyConfig


class ExtensionPolicyMixin:
    """Policy execution and timeline/physics callback behavior for Extension."""

    def _on_policy_button_clicked(self) -> None:
        """Start or stop the current robomimic policy chain.

        There are no inputs or outputs. This exists to share one button between
        starting queued policies and stopping the active controller.
        """

        if self._policy_controller is not None:
            self._stop_policy_execution("Robomimic policy chain stopped.")
            self._refresh_buttons_state()
            return

        if self._scene is None:
            self._set_status("Load the scene first.")
            return

        try:
            if self._queued_policy_configs:
                queue_to_run = list(self._queued_policy_configs)
            else:
                queue_to_run = [self._build_policy_config_from_models()]
        except Exception as exc:
            self._set_status(f"Failed to prepare policy chain: {exc}")
            self._refresh_buttons_state()
            return

        self._start_policy_queue(queue_to_run)

    def _on_timeline_event(self, event) -> None:
        """Handle timeline play/stop events and physics-step subscription.

        Input is a timeline event; there is no output.
        This exists so policy stepping is subscribed only while simulation plays.
        """

        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if self._physx_subscription is None:
                self._physx_subscription = self._physx_iface.subscribe_physics_step_events(self._on_physics_step)
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._physx_subscription = None
            if self._policy_controller is not None:
                try:
                    self._policy_controller.reset()
                except Exception:
                    self._stop_policy_execution("Robomimic policy chain stopped after a reset.")
        self._refresh_buttons_state()

    def _on_physics_step(self, step: float) -> None:
        """Advance the active policy controller on each PhysX step.

        Input is the PhysX step duration; there is no output.
        This exists to bridge simulation callbacks to the policy controller and
        advance queued policies when tasks complete.
        """

        if self._policy_controller is None:
            return

        try:
            self._policy_controller.on_physics_step(step)
            if self._policy_controller.is_task_complete():
                self._advance_to_next_policy(self._policy_controller.completion_status_message)
        except Exception as exc:
            self._stop_policy_execution(f"Robomimic policy error: {exc}")
            self._refresh_buttons_state()

    def _start_policy_queue(self, queue_to_run: list[QueuedPolicyConfig]) -> None:
        """Initialize queue execution state and start the first policy.

        Input is a list of queued policy configs; there is no output.
        This exists to separate queue setup from the UI button handler.
        """

        if not queue_to_run:
            self._set_status("Add a compatible policy to the queue before starting.")
            self._refresh_buttons_state()
            return

        self._active_policy_queue = list(queue_to_run)
        self._active_policy_queue_index = -1
        self._start_next_policy(previous_status_message=None)

    def _advance_to_next_policy(self, completion_status_message: str | None) -> None:
        """Stop the completed controller and continue or finish the queue.

        Input is an optional completion message; there is no output.
        This exists to make sequential policy execution robust after each task
        completion event.
        """

        completed_policy_index = self._active_policy_queue_index
        completed_queue_length = len(self._active_policy_queue)
        self._stop_active_policy_controller()
        if completed_queue_length <= 0:
            self._stop_policy_execution(completion_status_message or "Policy is complete.")
            self._refresh_buttons_state()
            return

        if completed_policy_index >= len(self._active_policy_queue) - 1:
            final_status_lines = []
            if completion_status_message:
                final_status_lines.append(completion_status_message)
            final_status_lines.append(
                f"Policy chain finished. Completed {completed_policy_index + 1} of {completed_queue_length} policies."
            )
            self._active_policy_queue = []
            self._active_policy_queue_index = -1
            self._set_status("\n".join(final_status_lines))
            self._refresh_buttons_state()
            return

        self._start_next_policy(previous_status_message=completion_status_message)

    def _start_next_policy(self, previous_status_message: str | None) -> None:
        """Start the next config in the active queue.

        Input is an optional previous status message; there is no output.
        This exists so initial start and queue advancement share one policy-start
        path and status formatting.
        """

        next_policy_index = self._active_policy_queue_index + 1
        if next_policy_index >= len(self._active_policy_queue):
            self._stop_policy_execution(previous_status_message or "Policy chain finished.")
            self._refresh_buttons_state()
            return

        policy_config = self._active_policy_queue[next_policy_index]
        try:
            self._start_policy_config(policy_config)
        except Exception as exc:
            self._stop_policy_execution(f"Failed to start queued policy {next_policy_index + 1}: {exc}")
            self._refresh_buttons_state()
            return

        self._active_policy_queue_index = next_policy_index

        observation_keys = ", ".join(self._policy_controller.policy_observation_keys)
        status_lines = []
        if previous_status_message:
            status_lines.append(previous_status_message)
        status_lines.extend(
            (
                f"Running policy {next_policy_index + 1} of {len(self._active_policy_queue)}.",
                f"Checkpoint: {policy_config.checkpoint_path}",
                f"Target object: {policy_config.target_object_prim_path}",
                f"Target position: {policy_config.target_position_prim_path}",
                f"Policy inputs: {observation_keys}",
            )
        )
        self._set_status("\n".join(status_lines))
        self._refresh_buttons_state()

    def _start_policy_config(self, policy_config: QueuedPolicyConfig) -> None:
        """Create a worker/controller for one queued policy config.

        Input is a queued policy config; there is no output.
        This exists to keep worker ownership, controller construction, and
        timeline startup in one failure-safe operation.
        """

        if self._scene is None:
            raise RuntimeError("Load the scene first.")

        self._stop_active_policy_controller()

        worker = None
        try:
            worker = RobomimicInferenceWorker(
                checkpoint_path=policy_config.checkpoint_path,
                norm_factor_min=policy_config.norm_factor_min,
                norm_factor_max=policy_config.norm_factor_max,
            )
            self._policy_controller = RobomimicPolicyController(
                scene=self._scene,
                worker=worker,
                target_object_prim_path=policy_config.target_object_prim_path,
                target_position_prim_path=policy_config.target_position_prim_path,
                status_callback=self._set_status,
            )
        except Exception:
            if worker is not None:
                worker.close()
            self._policy_controller = None
            raise

        if not self._timeline.is_playing():
            self._timeline.play()

    def _stop_active_policy_controller(self) -> None:
        """Hold and close the currently active policy controller.

        There are no inputs or outputs. This exists to make stop paths idempotent
        and safe even if hold or close fails during shutdown.
        """

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

    def _stop_policy_execution(self, status_message: str | None = None) -> None:
        """Stop any active controller and clear active queue state.

        Input is an optional status message; there is no output.
        This exists as the shared cleanup path for shutdown, scene reload, errors,
        and explicit user stop.
        """

        self._stop_active_policy_controller()
        self._active_policy_queue = []
        self._active_policy_queue_index = -1
        if status_message is not None:
            self._set_status(status_message)
