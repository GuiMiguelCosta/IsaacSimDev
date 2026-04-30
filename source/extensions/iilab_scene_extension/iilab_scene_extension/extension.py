from __future__ import annotations

from pathlib import Path

import omni.ext
import omni.physx
import omni.timeline

from .constants import DEFAULT_ROBOMIMIC_CHECKPOINT
from .extension_policy import ExtensionPolicyMixin
from .extension_queue import ExtensionQueueMixin
from .extension_scene import ExtensionSceneMixin
from .extension_ui import ExtensionUiMixin
from .policy import RobomimicPolicyController, guess_latest_checkpoint
from .policy_config import QueuedPolicyConfig
from .queue_config import PolicyQueueConfigStore
from .scene_model import IILABScene


class Extension(ExtensionSceneMixin, ExtensionQueueMixin, ExtensionPolicyMixin, ExtensionUiMixin, omni.ext.IExt):
    """Kit extension entry point composed from responsibility-focused mixins."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize extension state, subscriptions, models, and window.

        Input is the extension id from Kit; there is no output.
        This exists as the top-level lifecycle entry point while responsibility
        for UI, queues, scene edits, and policy execution lives in mixins.
        """

        self._ext_id = ext_id
        self._timeline = omni.timeline.get_timeline_interface()
        self._physx_iface = omni.physx.get_physx_interface()
        self._scene: IILABScene | None = None
        self._policy_controller: RobomimicPolicyController | None = None
        self._queued_policy_configs: list[QueuedPolicyConfig] = []
        self._active_policy_queue: list[QueuedPolicyConfig] = []
        self._active_policy_queue_index = -1
        self._is_loading = False
        self._physx_subscription = None
        self._timeline_subscription = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )
        self._queue_config_store = PolicyQueueConfigStore()

        default_checkpoint = DEFAULT_ROBOMIMIC_CHECKPOINT or guess_latest_checkpoint()
        default_policies_folder = str(Path(default_checkpoint).expanduser().parent) if default_checkpoint else ""

        self._initialize_ui_models(default_policies_folder)
        self._build_window()
        self._update_queue_summary()
        self._refresh_object_controls()
        self._refresh_available_policy_controls()
        self._refresh_buttons_state()

    def on_shutdown(self) -> None:
        """Release policy, scene, subscription, and window resources.

        There are no inputs or outputs. This exists as the Kit lifecycle cleanup
        point and delegates controller-specific cleanup to policy helpers.
        """

        self._stop_policy_execution()
        self._scene = None
        self._timeline_subscription = None
        self._physx_subscription = None
        self._window = None
