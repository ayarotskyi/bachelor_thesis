import inputs
import threading
import time

class GamepadListener:
    def __init__(self):
        self.running = False
        self.gamepad = None

    def find_gamepad(self):
        """Find the first available gamepad."""
        gamepads = inputs.devices.gamepads
        print(inputs.devices.all_devices)
        if len(gamepads) > 0:
            return gamepads[0]
        else:
            return None

    def start(self):
        """Start listening for gamepad events."""
        self.gamepad = self.find_gamepad()
        if not self.gamepad:
            print("No gamepad found. Please connect a gamepad and try again.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._listen_for_events)
        self.thread.start()

    def stop(self):
        """Stop listening for gamepad events."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _listen_for_events(self):
        """Listen for gamepad events and print them."""
        while self.running:
            try:
                events = self.gamepad.read()
                for event in events:
                    self._print_event(event)
            except inputs.UnpluggedError:
                print("Gamepad disconnected")
                self.running = False
            except Exception as e:
                print(f"Error reading gamepad: {e}")
                time.sleep(1)  # Avoid busy-waiting if there's a persistent error

    def _print_event(self, event):
        """Print gamepad event information."""
        if event.ev_type == 'Key':
            print(f"Button {event.code} {'pressed' if event.state == 1 else 'released'}")
        elif event.ev_type == 'Absolute':
            if event.code.startswith('ABS_'):
                print(f"Axis {event.code[4:]} value: {event.state}")

if __name__ == "__main__":
    listener = GamepadListener()
    listener.start()

    print("Listening for gamepad events. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping gamepad listener...")
        listener.stop()
        print("Gamepad listener stopped.")