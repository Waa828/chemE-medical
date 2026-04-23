"""
MKS Servo Motor USB/Serial Control
-----------------------------------
Provides two classes:
  - MKSMotor  : low-level Modbus RTU driver for MKS servo motors.
  - MotorControlUI : Jupyter ipywidgets UI that wraps MKSMotor.

Typical usage in a Jupyter notebook:
    motor = MKSMotor(port="COM3", unit=1)
    ui = MotorControlUI(motor)
    ui.display_controls()
"""

from pymodbus.client import ModbusSerialClient
import time
import threading
import ipywidgets as widgets
from IPython.display import display
import os
from datetime import datetime
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Modbus register addresses (MKS servo protocol)
# ---------------------------------------------------------------------------
_REG_HOMING_CMD    = 0x0091  # Write 1 to trigger homing
_REG_HOMING_STATUS = 0x00F1  # Read: 1 = homing complete
_REG_SPEED_CMD     = 0x00F6  # Write [dir|acc, speed] for continuous speed mode
_REG_ESTOP         = 0x00F7  # Write 1 for emergency stop
_REG_ENCODER       = 0x0031  # 3 × 16-bit registers encoding a 48-bit signed position

# Maps human-readable direction names to the hardware bit the protocol expects.
_DIRECTION_BITS = {"forward": 0, "backward": 1}

class MKSMotor:
    """Low-level driver for an MKS servo motor via Modbus RTU over USB/serial."""

    def __init__(
        self,
        port,
        baudrate=38400,
        parity="N",
        stopbits=1,
        bytesize=8,
        timeout=1,
        unit=1,
        max_range=315,
    ):
        """
        :param port:      Serial port identifier, e.g. "COM3" or "/dev/ttyUSB0".
        :param baudrate:  Baud rate (must match motor DIP-switch setting).
        :param parity:    Parity: 'N' none, 'E' even, 'O' odd.
        :param stopbits:  Number of stop bits.
        :param bytesize:  Number of data bits.
        :param timeout:   Read timeout in seconds.
        :param unit:      Modbus slave address of the motor.
        :param max_range: Software travel limit in mm (not yet enforced in hardware).
        """
        self.client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stopbits=stopbits,
            bytesize=bytesize,
            timeout=timeout,
            handle_local_echo=True,
            retries=3,
        )
        self.unit = unit
        self.max_range = max_range

        # Set by go_home(); used to compute physical position relative to home.
        self.encoder_pos_zero = None
        self.have_go_home = False

        # Signalled by emergency_stop() so that a running safe_run_motor() loop
        # can detect an external stop and exit cleanly.
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        """Open the serial connection to the motor."""
        self.client.connect()

    def close(self):
        """Close the serial connection, interrupting any running safe_run_motor() loop first."""
        self._stop_event.set()  # Signal safe_run_motor() to exit before the port closes
        time.sleep(0.1)         # Give the polling thread one cycle to see the flag
        self.client.close()

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------

    def go_home(self, time_out=60):
        """
        Trigger the motor's built-in homing routine and wait for it to finish.

        After a successful homing the current encoder tick is stored as the
        zero reference so that read_physical_position() can return mm values.

        :param time_out: Maximum seconds to wait before raising TimeoutError.
        :raises RuntimeError: Wraps any underlying communication or timeout error.
        """
        self.have_go_home = False
        self._stop_event.set()  # Interrupt any running safe_run_motor() loop before homing

        try:
            response = self.client.write_register(_REG_HOMING_CMD, 1, device_id=self.unit)
            if response.isError():
                raise ValueError("Failed to send homing command")

            print("Homing started…")
            start_time = time.time()
            time.sleep(1)  # Brief pause before polling so the motor has time to start moving

            while True:
                time.sleep(0.1)
                elapsed = time.time() - start_time
                if elapsed > time_out:
                    raise TimeoutError(
                        f"Homing timed out after {time_out} seconds"
                    )

                status_resp = self.client.read_input_registers(
                    address=_REG_HOMING_STATUS, count=1, device_id=self.unit
                )
                if status_resp.isError():
                    print("Warning: failed to read homing status — retrying")
                    continue

                # Register value 1 signals that homing is complete
                if status_resp.registers[0] == 1:
                    print("Homing complete")
                    break

            time.sleep(1)  # Let the motor settle before reading the zero position
            self.have_go_home = True
            self.encoder_pos_zero = self.read_encoder_position()
            return "Motor homing succeeded"

        except Exception as e:
            raise RuntimeError(f"Error during homing: {e}") from e

    def emergency_stop(self):
        """
        Immediately halt the motor via the emergency-stop register.

        Also sets the internal stop-event so that any concurrently running
        safe_run_motor() loop detects the interruption and exits cleanly.
        """
        self._stop_event.set()  # Signal safe_run_motor() to break out of its loop
        self.client.write_register(_REG_ESTOP, 1, device_id=self.unit)
        time.sleep(0.1)
        return "Motor emergency stop triggered"

    def run_motor(self, dire, acc, speed):
        """
        Command the motor to run continuously in speed mode.

        :param dire:  Direction — ``"forward"`` or ``"backward"``.
        :param acc:   Acceleration value (0–255).
        :param speed: Target speed (motor units).
        :raises ValueError:  If ``dire`` is not ``"forward"`` or ``"backward"``.
        :raises RuntimeError: Prints a warning and returns early if homing has not been done.
        """
        if not self.have_go_home:
            print("Motor has not been homed — run go_home() first")
            return

        if dire not in _DIRECTION_BITS:
            raise ValueError(f"dire must be 'forward' or 'backward', got {dire!r}")

        # The protocol packs direction and acceleration into a single 16-bit word:
        # high byte = direction bit (0=forward, 1=backward), low byte = acceleration.
        dir_bit = _DIRECTION_BITS[dire]
        values = [dir_bit * 256 + acc, speed]
        self.client.write_registers(_REG_SPEED_CMD, values, device_id=self.unit)
        return f"Motor #{self.unit} running — direction: {dire}, acc: {acc}, speed: {speed}"

    def safe_run_motor(self, dire, acc, speed, poll_interval=0.05):
        """
        Run the motor continuously but stop automatically at the travel limits.

        - ``"forward"``:  polls position until pos >= max_range, then stops.
        - ``"backward"``: polls position until pos <= 0 mm (home), then stops.

        This is a **blocking** call.  Wrap it in a ``threading.Thread`` when
        calling from a GUI so the UI stays responsive (see MotorControlUI).

        :param dire:          Direction — ``"forward"`` or ``"backward"``.
        :param acc:           Acceleration value (0–255).
        :param speed:         Target speed (motor units).
        :param poll_interval: Seconds between position polls (default 0.05 s).
        :return:              Message string describing where the motor stopped.
        :raises RuntimeError: If a communication error occurs; motor is stopped first.
        """
        if not self.have_go_home:
            print("Motor has not been homed — run go_home() first")
            return

        self._stop_event.clear()  # Reset any leftover signal from a previous stop
        self.run_motor(dire, acc, speed)

        try:
            while True:
                time.sleep(poll_interval)

                # Emergency stop was pressed externally — exit without calling
                # emergency_stop() again (it has already been sent to the motor).
                if self._stop_event.is_set():
                    return "Safe run interrupted by emergency stop or disconnection."

                pos = self.read_physical_position()

                if pos is None:
                    continue  # Transient read failure — keep polling

                if dire == "forward" and pos >= self.max_range:
                    self.emergency_stop()
                    return f"Forward limit reached — stopped at {pos:.3f} mm"

                if dire == "backward" and pos <= 0:
                    self.emergency_stop()
                    return f"Home limit reached — stopped at {pos:.3f} mm"

        except Exception as e:
            # Always stop the motor before propagating any error
            self.emergency_stop()
            raise RuntimeError(f"Error in safe_run_motor: {e}") from e

    # ------------------------------------------------------------------
    # Position readback
    # ------------------------------------------------------------------

    def read_encoder_position(self, register=_REG_ENCODER, count=3):
        """
        Read the raw 48-bit signed encoder tick count from the motor.

        The motor stores the position across three consecutive 16-bit Modbus
        input registers (big-endian, most-significant word first).

        :param register: Starting register address (default: _REG_ENCODER).
        :param count:    Number of 16-bit registers to read (must be 3).
        :return:         Signed integer tick count, or None on communication error.
        """
        result = self.client.read_input_registers(
            address=register, count=count, device_id=self.unit
        )
        if result.isError():
            print(f"Encoder read failed: {result}")
            return None

        # Reconstruct the 48-bit value from three 16-bit words
        position = (
            (result.registers[0] << 32)
            | (result.registers[1] << 16)
            | result.registers[2]
        )

        # Interpret as a signed 48-bit integer (two's complement)
        if position >= 2**47:
            position -= 2**48

        return position

    def read_physical_position(self):
        """
        Return the current axis position in millimetres relative to the home position.

        Conversion: the motor encoder has 16 384 ticks per revolution and the
        lead-screw pitch is 40 mm/rev, so:
            position_mm = -Δticks / 16384 * 40

        The sign is negated so that positive values correspond to forward travel.

        :return: Position in mm (float), or None if homing has not been performed.
        """
        if not self.have_go_home:
            print("Go home first!")
            return None

        current = self.read_encoder_position()
        position_mm = -(current - self.encoder_pos_zero) / 16384 * 40
        return position_mm


# ---------------------------------------------------------------------------
# Jupyter widget UI
# ---------------------------------------------------------------------------

class MotorControlUI:
    """
    Interactive ipywidgets panel for controlling an MKSMotor instance.

    Call display_controls() inside a Jupyter notebook cell to render the UI.
    """

    def __init__(self, motor: MKSMotor):
        """
        :param motor: An MKSMotor instance (need not be connected yet).
        """
        self.motor = motor
        self.output_area = widgets.Output()

        # --- Connection buttons ---
        self.connect_button       = widgets.Button(description="Connect")
        self.close_button         = widgets.Button(description="Disconnect")

        # --- Motion control buttons ---
        self.home_button           = widgets.Button(description="Go Home")
        self.emergency_stop_button = widgets.Button(description="Emergency Stop")
        self.run_motor_button      = widgets.Button(description="Run Motor")
        self.safe_run_motor_button = widgets.Button(
            description="Safe Run",
            button_style="success",  # Green to distinguish it from the raw run
            tooltip=f"Run with automatic stop at limits (0 – {motor.max_range} mm)",
        )
        self.read_position_button  = widgets.Button(description="Read Position")

        # --- Parameter inputs ---
        self.dire_input  = widgets.Dropdown(
            options=["forward", "backward"],
            value="forward",
            description="Direction",
        )
        self.acc_input   = widgets.IntText(value=255, description="Acc")
        self.speed_input = widgets.IntText(value=5,   description="Speed")

        # --- Position display ---
        self.position_label = widgets.Label(value="Position: --")

        # Wire up callbacks
        self.connect_button.on_click(self._connect_motor)
        self.close_button.on_click(self._close_motor)
        self.home_button.on_click(self._go_home)
        self.emergency_stop_button.on_click(self._emergency_stop)
        self.run_motor_button.on_click(self._run_motor)
        self.safe_run_motor_button.on_click(self._safe_run_motor)
        self.read_position_button.on_click(self._read_position)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message):
        """Display a message in the output area, replacing any previous content."""
        with self.output_area:
            self.output_area.clear_output(wait=True)
            print(message)

    # ------------------------------------------------------------------
    # Button callbacks  (each takes the click event argument '_')
    # ------------------------------------------------------------------

    def _connect_motor(self, _):
        try:
            self.motor.connect()
            self._log("Connected to motor")
        except Exception as e:
            self._log(f"Error connecting: {e}")

    def _close_motor(self, _):
        try:
            self.motor.close()
            self._log("Connection closed")
        except Exception as e:
            self._log(f"Error closing connection: {e}")

    def _go_home(self, _):
        try:
            result = self.motor.go_home()
            self._log(result)
        except Exception as e:
            self._log(f"Error during homing: {e}")

    def _emergency_stop(self, _):
        try:
            result = self.motor.emergency_stop()
            self._log(result)
        except Exception as e:
            self._log(f"Error with emergency stop: {e}")

    def _run_motor(self, _):
        try:
            result = self.motor.run_motor(
                self.dire_input.value,
                self.acc_input.value,
                self.speed_input.value,
            )
            self._log(result)
        except Exception as e:
            self._log(f"Error running motor: {e}")

    def _safe_run_motor(self, _):
        """
        Launch safe_run_motor() in a background thread so the UI stays responsive.

        The button is disabled while the motor is moving and re-enabled once
        the limit is reached (or an error occurs).
        """
        dire  = self.dire_input.value
        acc   = self.acc_input.value
        speed = self.speed_input.value

        def worker():
            self.safe_run_motor_button.disabled = True
            self._log(
                f"Safe run started — direction: {dire}, acc: {acc}, speed: {speed}"
            )
            try:
                result = self.motor.safe_run_motor(dire, acc, speed)
                self._log(result)
                # Refresh the position label after stopping
                pos = self.motor.read_physical_position()
                if pos is not None:
                    self.position_label.value = f"Position: {pos:.3f}"
            except Exception as e:
                self._log(f"Error in safe run: {e}")
            finally:
                self.safe_run_motor_button.disabled = False

        threading.Thread(target=worker, daemon=True).start()

    def _read_position(self, _):
        try:
            pos = self.motor.read_physical_position()
            self._log(f"Motor position: {pos:.3f} mm")
            self.position_label.value = f"Position: {pos:.3f}"
        except Exception as e:
            self._log(f"Error reading position: {e}")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def display_controls(self):
        """Render the full motor-control panel in the current notebook cell."""
        col1 = widgets.VBox([
            widgets.HTML("<b>Connection</b>"),
            self.connect_button,
            self.close_button,
        ])

        col2 = widgets.VBox([
            widgets.HTML("<b>Motion controls</b>"),
            self.home_button,
            self.emergency_stop_button,
            # self.run_motor_button,
            self.safe_run_motor_button,
            self.read_position_button,
        ])

        col3 = widgets.VBox([
            widgets.HTML("<b>Parameters &amp; position</b>"),
            self.dire_input,
            self.acc_input,
            self.speed_input,
            widgets.HBox([self.position_label, widgets.HTML("&nbsp;mm")]),
        ])

        display(widgets.HBox([col1, col2, col3]), self.output_area)
