import signal

class CtrlCHandler:
    """
    Handle Ctrl+C for clean shutdown to avoid data corruption
    and abrupt device disconnect.
    """
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Callback executed when Ctrl+C is detected"""
        print("\n[INFO] Interrupt signal detected, closing cleanly...")
        self.should_stop = True