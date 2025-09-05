import os.path
from pathlib import Path

from kivy.properties import StringProperty, BooleanProperty
from kivymd.uix.card import MDCard

from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from datetime import datetime, time, timezone

from kivy.uix.behaviors import FocusBehavior

# TODO: ORGANIZE CODE


class TimePickerContent(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type('on_kv_post')  # ensure KV is loaded before accessing `ids`

    def on_kv_post(self, base_widget):
        for field in [self.ids.hour, self.ids.minute, self.ids.second]:
            field.bind(focus=self._select_all_on_focus)

    def _select_all_on_focus(self, instance, value):
        if value:  # If gaining focus
            # Delay slightly to allow focus to settle before selecting
            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: instance.select_all())


class ModeCard(MDCard):
    mode_key = StringProperty()
    icon = StringProperty()
    title = StringProperty()
    description = StringProperty()
    selected = BooleanProperty(False)
    hovered = BooleanProperty(False)

    def on_hover_enter(self):
        self.hovered = True
        app = MDApp.get_running_app()
        app.focus_index_mode = app.mode_card_list.index(self)  # Dynamically set focus based on the card hovered
        app.set_mode_card_focus(self)  # Focus the current hovered card

    def on_hover_leave(self):
        self.hovered = False
        self.canvas.ask_update()  # Force update


# Extend ModeCard to be focusable and respond to keyboard
class FocusableModeCard(FocusBehavior, ModeCard):
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] == "enter":
            app = MDApp.get_running_app()
            app.select_mode(self.mode_key)
            return True
        return super().keyboard_on_key_down(window, keycode, text, modifiers)


class ModeApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_mode = None
        self.selected_start_datetime = None
        self.selected_end_datetime = None
        self.selected_date = datetime.today().date()
        self.local_timezone = datetime.now().astimezone().tzinfo
        self.focus_index = 0  # For time picker focus cycling
        self.focusables = []

        self.picking_end_time = False  # Track if picking end time

        self.mode_card_list = []
        self.focus_index_mode = 0  # For mode cards keyboard focus cycling

    def build(self):
        Window.bind(on_key_down=self.on_key_down)
        root = Builder.load_file(os.path.join(Path(__file__).parent, "date_range.kv"))

        modes = [
            {"key": "realtime", "icon": "timer-sand", "title": "Real-time",
             "description": "Use current time automatically"},
            {"key": "since", "icon": "calendar-clock", "title": "Since",
             "description": "Select a start timestamp only"},
            {"key": "offline", "icon": "calendar-range", "title": "Offline",
             "description": "Select start and end timestamps"},
        ]

        mode_grid = root.ids.mode_grid
        self.mode_cards = {}
        self.mode_card_list = []

        for mode in modes:
            card = FocusableModeCard(
                mode_key=mode["key"],
                icon=mode["icon"],
                title=mode["title"],
                description=mode["description"],
                size_hint=(1, None),
                height="150dp",
            )
            mode_grid.add_widget(card)
            self.mode_cards[mode["key"]] = card
            self.mode_card_list.append(card)

        # Highlight/focus Realtime but do NOT select it yet
        self.focus_index_mode = 0
        self.set_mode_card_focus(self.mode_card_list[self.focus_index_mode])

        return root

    def set_mode_card_focus(self, card):
        for c in self.mode_card_list:
            c.selected = False  # deselect all
            c.focus = False
        card.focus = True
        card.selected = True

    def select_mode(self, mode_key):
        if self.selected_mode:
            return

        for card in self.mode_cards.values():
            card.selected = False

        selected_card = self.mode_cards.get(mode_key)
        if selected_card:
            selected_card.selected = True
        if mode_key == "offline":
            self.selected_mode = "offline"
            self.show_datetime_picker(picking_end=False)
        elif mode_key == "since":
            self.selected_mode = "since"
            self.show_datetime_picker(picking_end=False)
        elif mode_key == "realtime":
            self.selected_mode = "realtime"
            self.selected_start_datetime = datetime.now(timezone.utc)
            self.stop()
            Window.close()
        else:
            raise ValueError("Invalid mode")

    def show_datetime_picker(self, *, picking_end=False):
        from kivymd.uix.pickers import MDDatePicker
        self.picking_end_time = picking_end

        title = "Select END Date" if picking_end else "Select START Date"
        self.date_dialog = MDDatePicker(title=title)
        self.date_dialog.bind(on_save=self.on_date_selected, on_cancel=self.on_cancel)
        self.date_dialog.open()

    def on_date_selected(self, instance, value, date_range):
        self.selected_date = value
        self.show_time_picker()

    def show_time_picker(self):
        self.time_picker_content = TimePickerContent()

        self.back_btn = MDFlatButton(text="BACK", on_release=self.back_to_calendar)
        self.cancel_btn = MDFlatButton(text="CANCEL", on_release=self.close_dialog)
        self.ok_btn = MDFlatButton(text="OK", on_release=self.on_time_selected)

        dialog_title = "Select END Time" if self.picking_end_time else "Select START Time"

        self.dialog = MDDialog(
            title=dialog_title,
            type="custom",
            content_cls=self.time_picker_content,
            buttons=[self.back_btn, self.cancel_btn, self.ok_btn],
        )
        self.dialog.open()

        ids = self.time_picker_content.ids
        self.focusables = [ids.hour, ids.minute, ids.second, self.ok_btn, self.cancel_btn, self.back_btn]
        self.focus_index = 0
        self.set_focus(self.focusables[self.focus_index])

    def back_to_calendar(self, *args):
        self.dialog.dismiss()
        self.show_datetime_picker(picking_end=self.picking_end_time)

    def set_focus(self, widget):
        for w in self.focusables:
            if hasattr(w, "focus"):
                w.focus = False
            elif hasattr(w, "md_bg_color"):
                w.md_bg_color = (1, 1, 1, 0)

        if hasattr(widget, "focus"):
            widget.focus = True
        elif hasattr(widget, "md_bg_color"):
            widget.md_bg_color = self.theme_cls.primary_light

    def close_dialog(self, *args):
        self.dialog.dismiss()
        self.stop()
        Window.close()

    def on_time_selected(self, *args):
        self.focus_index = 0
        self.set_focus(self.focusables[self.focus_index])
        ids = self.time_picker_content.ids
        h, m, s = ids.hour.text, ids.minute.text, ids.second.text

        try:
            h, m, s = int(h), int(m), int(s)
            assert 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60
        except (ValueError, AssertionError):
            ids.hour.error = ids.minute.error = ids.second.error = True
            return

        self.dialog.dismiss()
        selected_time = time(h, m, s)
        # selected_time = selected_time.astimezone(ZoneInfo("UTC"))
        #
        if self.selected_mode == "offline" and not self.picking_end_time:
            # User selected START datetime
            self.selected_start_datetime = datetime.combine(self.selected_date, selected_time, tzinfo=self.local_timezone).astimezone(timezone.utc)
            # Prompt for END datetime

            self.show_datetime_picker(picking_end=True)

        elif self.selected_mode == "offline" and self.picking_end_time:
            # User selected END datetime
            self.selected_end_datetime = datetime.combine(self.selected_date, selected_time, tzinfo=self.local_timezone).astimezone(timezone.utc)

            # Validate end > start
            if self.selected_end_datetime <= self.selected_start_datetime:
                from kivymd.toast import toast
                toast("End datetime must be after Start datetime!")
                self.selected_end_datetime = None
                self.show_datetime_picker(picking_end=True)
                return

            self.root.ids.datetime_label.text = (
                f"Selected start: {self.selected_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Selected end: {self.selected_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.stop()
            Window.close()

        else:
            # For other modes: single datetime
            self.selected_start_datetime = datetime.combine(self.selected_date, selected_time, tzinfo=self.local_timezone).astimezone(timezone.utc)
            self.root.ids.datetime_label.text = f"Selected datetime:\n{self.selected_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            self.stop()
            Window.close()

    def on_cancel(self, *args):
        self.selected_start_datetime = None
        self.selected_end_datetime = None
        self.stop()
        Window.close()

    def on_key_down(self, window, key, scancode, codepoint, modifiers):
        # Keyboard navigation for mode cards (only if dialog not open)
        if not hasattr(self, 'dialog') or not self.dialog or not self.dialog.open:
            if hasattr(self, 'mode_card_list') and self.mode_card_list:
                if key == 9:  # Tab
                    if "shift" in modifiers:
                        self.focus_index_mode = (self.focus_index_mode - 1) % len(self.mode_card_list)
                    else:
                        self.focus_index_mode = (self.focus_index_mode + 1) % len(self.mode_card_list)
                    self.set_mode_card_focus(self.mode_card_list[self.focus_index_mode])
                    return True

                elif key == 13:  # Enter
                    focused_card = self.mode_card_list[self.focus_index_mode]
                    self.select_mode(focused_card.mode_key)
                    return True

        # Keyboard navigation inside dialog
        if hasattr(self, 'focusables') and self.focusables:
            if key == 9:  # Tab key
                if "shift" in modifiers:
                    self.focus_index = (self.focus_index - 1) % len(self.focusables)
                else:
                    self.focus_index = (self.focus_index + 1) % len(self.focusables)
                self.set_focus(self.focusables[self.focus_index])
                return True

            elif key == 13:  # Enter key
                current = self.focusables[self.focus_index]
                if current == self.ok_btn:
                    self.on_time_selected()
                elif current == self.cancel_btn:
                    self.close_dialog()
                elif current == self.back_btn:
                    self.back_to_calendar()
                return True

        return False


if __name__ == "__main__":
    app = ModeApp()
    app.run()

    print("Selected mode:", app.selected_mode)
    print("Start datetime:", app.selected_start_datetime)
    print("End datetime:", app.selected_end_datetime)
