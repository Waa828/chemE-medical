"""
contest_helper_code.py
----------------------
Reusable machinery for the AI-driven ChemE Car competition notebook.

Public API
----------
preflight_check(cam_control, motor, mnist_model=..., step=None)
    Raise RuntimeError with friendly messages if the system is not ready.

DigitContestRunner(motor, cam_control, motor_ui, mnist_model)
    Step 2 – run motor until the camera sees the target MNIST digit.
    .display()      → renders the widget UI in the notebook cell
    .stop_pos       → final motor position (mm) after the contest, or None
    .stop_elapsed   → elapsed time (s) when the motor stopped, or None

ColorContestRunner(motor, cam_control, motor_ui)
    Step 4 – run motor until the color camera detects the iodine-clock colour change.
    Same .display() / .stop_pos / .stop_elapsed interface.
"""

import time
import threading
import ipywidgets as widgets
import os
from IPython.display import display as _display

# ── Sentinel used by preflight_check to distinguish "not provided" from None ──
_NOT_PROVIDED = object()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight check
# ─────────────────────────────────────────────────────────────────────────────

def preflight_check(cam_control, motor, mnist_model=_NOT_PROVIDED, step=None):
    """
    Verify that cameras, motor, and (optionally) MNIST model are ready.

    Parameters
    ----------
    cam_control : CameraControl
    motor       : MKSMotor
    mnist_model : model object with .predict(), or the default sentinel to skip
                  the model check (Step 4), or None if the variable is undefined.
    step        : int | None  – used only in the printed error header.

    Raises
    ------
    RuntimeError if any check fails (stops further cell execution).
    """
    errors = []
    step_label = f"Step {step}" if step else "Contest"

    if not cam_control.is_running:
        errors.append(
            "Cameras are not running — click 'Start Cameras' in the setup cell."
        )

    try:
        motor_ok = motor.client.is_socket_open()
    except Exception:
        motor_ok = False
    if not motor_ok:
        errors.append(
            "Motor is not connected — click 'Connect' in the motor control UI."
        )

    # mnist_model=_NOT_PROVIDED → caller chose not to check (Step 4)
    # mnist_model=None          → globals().get() returned None (undefined in notebook)
    if mnist_model is not _NOT_PROVIDED:
        model_ok = (
            mnist_model is not None
            and callable(getattr(mnist_model, "predict", None))
        )
        if not model_ok:
            errors.append("MNIST model is not loaded — run Step 1 first.")

    if errors:
        print(f"⚠️  Cannot start {step_label}. Fix the following issues:\n")
        for e in errors:
            print(f"  ❌ {e}")
        raise RuntimeError("Pre-flight check failed — see messages above.")


# ─────────────────────────────────────────────────────────────────────────────
# Shared base class
# ─────────────────────────────────────────────────────────────────────────────

class _ContestBase:
    """
    Common widget layout, threading, and motor helpers shared by both runners.
    Subclasses implement _contest_loop() and optional extra widget hooks.
    """

    # Physical constants (mm)
    _TRACK_MM   = 320
    _LEAD_PITCH = 40   # mm per motor revolution

    # Status messages – overridden by subclasses
    _success_msg = "Status: ✓ Done!"
    _fail_msg    = "Status: ✗ Not detected"

    def __init__(self, motor, cam_control, motor_ui):
        self._motor    = motor
        self._cam      = cam_control
        self._motor_ui = motor_ui
        self._running  = False

        self.stop_elapsed = None
        self.stop_pos     = None

        # ── Shared widgets ────────────────────────────────────────────────────
        self._status_lbl = widgets.Label(value="Status: Idle")
        self._time_lbl   = widgets.Label(value="Elapsed Time: – s")
        self._pos_lbl    = widgets.Label(value="Stop Position: – mm")
        self._output     = widgets.Output()
        self._run_btn    = widgets.Button(
            description="Run Contest", button_style="success"
        )
        self._stop_btn   = widgets.Button(
            description="Stop", button_style="danger", disabled=True
        )
        self._run_btn.on_click(self._on_run)
        self._stop_btn.on_click(self._on_stop)

    # ── Motor helpers ─────────────────────────────────────────────────────────

    def _home_motor(self):
        self._status_lbl.value = "Status: Homing motor…"
        with self._output:
            print("Homing motor…")
        self._motor.emergency_stop()
        self._motor.go_home()

    def _motor_params(self):
        return (
            self._motor_ui.dire_input.value,
            self._motor_ui.acc_input.value,
            self._motor_ui.speed_input.value,
        )

    def _max_time(self, spd):
        # Formula: track_length(mm) / lead_pitch(mm/rev) / speed(rpm) * 60(s/min)
        return self._TRACK_MM / self._LEAD_PITCH / spd * 60

    def _safe_stop(self):
        """Emergency-stop motor and return final position; handles serial errors."""
        try:
            self._motor.emergency_stop()
            return self._motor.read_physical_position()
        except Exception as e:
            with self._output:
                print(f"Motor error during stop: {e}")
            return float("nan")

    # ── State management ──────────────────────────────────────────────────────

    def _finish(self, success: bool, elapsed: float, pos: float):
        self._cam.stop_recording()
        self._running         = False
        self._run_btn.disabled  = False
        self._stop_btn.disabled = True
        self.stop_elapsed     = elapsed
        self.stop_pos         = pos
        self._time_lbl.value  = f"Elapsed Time: {elapsed:.2f} s"
        self._pos_lbl.value   = f"Stop Position: {pos:.2f} mm"
        self._status_lbl.value = self._success_msg if success else self._fail_msg

    def _reset_labels(self):
        self._time_lbl.value   = "Elapsed Time: – s"
        self._pos_lbl.value    = "Stop Position: – mm"
        self._status_lbl.value = "Status: Idle"
        self._on_reset()        # subclass hook for extra labels

    def _on_reset(self):
        """Subclass hook: reset any additional labels."""

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_run(self, _):
        if self._running:
            return
        self._running           = True
        self._run_btn.disabled  = True
        self._stop_btn.disabled = False
        self._output.clear_output(wait=True)
        self._reset_labels()
        self._status_lbl.value = "Status: Starting…"
        threading.Thread(target=self._contest_loop, daemon=True).start()

    def _on_stop(self, _):
        self._running          = False   # signals the loop to exit on next iteration
        self._status_lbl.value = "Status: Stopping…"

    # ── Subclass interface ────────────────────────────────────────────────────

    def _contest_loop(self):
        raise NotImplementedError

    def _extra_left_widgets(self):
        """Widgets inserted above the Run/Stop buttons (e.g. target digit input)."""
        return []

    def _extra_right_widgets(self):
        """Widgets inserted between the header and the time/pos labels."""
        return []

    # ── Layout & display ──────────────────────────────────────────────────────

    def display(self):
        left_col = widgets.VBox([
            widgets.HTML("<b>Contest Controls</b>"),
            *self._extra_left_widgets(),
            widgets.HBox([self._run_btn, self._stop_btn]),
        ])
        right_col = widgets.VBox([
            widgets.HTML("<b>Contest Status</b>"),
            self._status_lbl,
            *self._extra_right_widgets(),
            self._time_lbl,
            self._pos_lbl,
        ])
        _display(widgets.VBox([widgets.HBox([left_col, right_col]), self._output]))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – digit recognition
# ─────────────────────────────────────────────────────────────────────────────

class DigitContestRunner(_ContestBase):
    """Run motor until the digit camera sees the target MNIST digit."""

    _success_msg = "Status: ✓ Target digit found!"
    _fail_msg    = "Status: ✗ Target digit not found"

    def __init__(self, motor, cam_control, motor_ui, mnist_model):
        super().__init__(motor, cam_control, motor_ui)
        self._model        = mnist_model
        self._target_input = widgets.IntText(value=7, description="Target Digit:")
        self._dot_lbl      = widgets.Label(value="Digit Card in View: No")
        self._pred_lbl     = widgets.Label(value="Predicted Digit: –")

    def _extra_left_widgets(self):
        return [self._target_input]

    def _extra_right_widgets(self):
        return [self._dot_lbl, self._pred_lbl]

    def _on_reset(self):
        self._dot_lbl.value  = "Digit Card in View: No"
        self._pred_lbl.value = "Predicted Digit: –"

    def _contest_loop(self):
        self._home_motor()

        dire, acc, spd = self._motor_params()
        max_time   = self._max_time(spd)
        run_result = self._motor.run_motor(dire, acc, spd)
        os.makedirs("video", exist_ok=True)
        video_filename = f"video/contest_{int(time.time()*1000)}.avi"
        self._cam.start_recording(video_filename)
        self._status_lbl.value = "Status: Running…"
        with self._output:
            self._output.clear_output(wait=True)
            print(f"Motor started : {run_result}")
            print(f"Target digit  : {self._target_input.value}")
            print(f"Max allowed time: {max_time:.1f} s")

        start_time = time.time()

        while self._running:
            elapsed = time.time() - start_time
            self._time_lbl.value = f"Elapsed Time: {elapsed:.2f} s"

            dot_in_center, _, nn_input_img = self._cam.get_data()
            self._dot_lbl.value = (
                f"Digit Card in View: {'Yes' if dot_in_center else 'No'}"
            )

            if dot_in_center:
                if nn_input_img is not None:
                    pred = self._model.predict(nn_input_img)
                    if pred == self._target_input.value:
                        self._motor.emergency_stop()
                        pos = self._motor.read_physical_position()
                        with self._output:
                            print(f"✓ Target digit {pred} detected → motor stopped.")
                            print(f"  Time: {elapsed:.2f} s  |  Position: {pos:.2f} mm")
                        self._finish(True, elapsed, pos)
                        return
                else:
                    self._pred_lbl.value = "Predicted Digit: (no image)"

            if elapsed > max_time:
                pos = self._safe_stop()
                with self._output:
                    print(f"✗ Max time reached – motor stopped at {pos:.2f} mm.")
                self._finish(False, elapsed, pos)
                return

            time.sleep(0.2)

        # Reached only when Stop button is pressed
        elapsed = time.time() - start_time
        with self._output:
            print("Contest manually stopped.")
        pos = self._safe_stop()
        self._finish(False, elapsed, pos)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – colour change detection
# ─────────────────────────────────────────────────────────────────────────────

class ColorContestRunner(_ContestBase):
    """Run motor until the colour camera detects the iodine-clock colour change."""

    _success_msg = "Status: ✓ Color change detected!"
    _fail_msg    = "Status: ✗ Color change not detected"

    def __init__(self, motor, cam_control, motor_ui):
        super().__init__(motor, cam_control, motor_ui)
        self._color_lbl = widgets.Label(value="Color Change Detected: No")

    def _extra_right_widgets(self):
        return [self._color_lbl]

    def _on_reset(self):
        self._color_lbl.value = "Color Change Detected: No"

    def _contest_loop(self):
        self._home_motor()

        dire, acc, spd = self._motor_params()
        max_time   = self._max_time(spd)
        run_result = self._motor.run_motor(dire, acc, spd)
        os.makedirs("video", exist_ok=True)
        video_filename = f"video/contest_{int(time.time()*1000)}.avi"
        self._cam.start_recording(video_filename)
        self._status_lbl.value = "Status: Running…"
        with self._output:
            self._output.clear_output(wait=True)
            print(f"Motor started : {run_result}")
            print(f"Max allowed time: {max_time:.1f} s")

        start_time = time.time()

        while self._running:
            elapsed = time.time() - start_time
            self._time_lbl.value = f"Elapsed Time: {elapsed:.2f} s"

            _, color_change, _ = self._cam.get_data()
            self._color_lbl.value = (
                f"Color Change Detected: {'Yes' if color_change else 'No'}"
            )

            if color_change:
                self._motor.emergency_stop()
                pos = self._motor.read_physical_position()
                with self._output:
                    print(f"✓ Color change detected → motor stopped.")
                    print(f"  Time: {elapsed:.2f} s  |  Position: {pos:.2f} mm")
                self._finish(True, elapsed, pos)
                return

            if elapsed > max_time:
                pos = self._safe_stop()
                with self._output:
                    print(f"✗ Max time reached – motor stopped at {pos:.2f} mm.")
                self._finish(False, elapsed, pos)
                return

            time.sleep(0.2)

        # Reached only when Stop button is pressed
        elapsed = time.time() - start_time
        with self._output:
            print("Contest manually stopped.")
        pos = self._safe_stop()
        self._finish(False, elapsed, pos)
