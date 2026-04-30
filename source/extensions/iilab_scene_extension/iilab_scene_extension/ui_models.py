from __future__ import annotations

import omni.ui as ui


class ComboBoxItem(ui.AbstractItem):
    """One text row item for an Omni UI combo box."""

    def __init__(self, text: str) -> None:
        """Create a single text item for an Omni UI combo box.

        Input is the displayed text; there is no return value.
        This exists to keep combo box row construction out of extension logic.
        """

        super().__init__()
        self.model = ui.SimpleStringModel(text)


class ComboBoxModel(ui.AbstractItemModel):
    """Flat string-list model for Omni UI combo boxes."""

    def __init__(self, items: list[str], selected_index: int = 0) -> None:
        """Create a combo model with a clamped initial selection.

        Inputs are the display items and preferred selection index; there is no
        return value. This exists because Omni UI requires an AbstractItemModel
        wrapper instead of accepting a plain list.
        """

        super().__init__()
        self._items = [ComboBoxItem(item) for item in items]
        self._current_index = ui.SimpleIntModel(self._clamp_index(selected_index))
        self._current_index.add_value_changed_fn(lambda _: self._item_changed(None))

    def _clamp_index(self, index: int) -> int:
        """Clamp a requested selection index to the available items.

        Input is an integer index; the output is a safe integer index.
        This exists to prevent stale UI selections from escaping list bounds.
        """

        if not self._items:
            return 0
        return max(0, min(index, len(self._items) - 1))

    @property
    def selected_index(self) -> int:
        """Return the current selection as a safe list index.

        There are no inputs; the output is an integer index.
        This exists so callers do not need to duplicate bounds checks.
        """

        return self._clamp_index(self._current_index.get_value_as_int())

    def get_item_children(self, item):
        """Return the combo box row items requested by Omni UI.

        Input is ignored because this model is flat; the output is the item list.
        This exists to satisfy AbstractItemModel's child enumeration contract.
        """

        return self._items

    def get_item_value_model(self, item: ui.AbstractItem = None, column_id: int = 0):
        """Return the value model for the selection or a row item.

        Inputs are the optional item and column id; the output is a UI model.
        This exists to bridge Omni UI's selection model with per-row text models.
        """

        if item is None:
            return self._current_index
        return item.model
