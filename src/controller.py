class SwitchController:
    def __init__(self, classifier):
        self.classifier = classifier
        
        # State machine variables
        self.mode = "execute"  # Modes: 'execute' or 'register'
        self.target_label = None
        self.taps_collected = 0
        self.taps_needed = 3

    def set_registration_mode(self, label: str, taps: int = 10):
        """
        Triggered by your Web UI. Switches the system into learning mode.
        """
        self.mode = "register"
        self.target_label = label
        self.taps_collected = 0
        self.taps_needed = taps
        print(f"\n[*] REGISTRATION MODE: Please tap your '{label}' {taps} times.")


    def set_execution_mode(self):
        self.mode = "execute"
        print(f"\n[*] EXECUTION MODE: Actively listening")

    def audio_callback(self, features, tap_count):
        """
        This is the exact function passed into listener.listen().
        It fires every time the microphone detects a knock.
        """
        if self.mode == "register":
            # --- ROUTE A: Teaching the system ---
            self.taps_collected += 1
            print(f"[*] Captured tap {self.taps_collected}/{self.taps_needed} for '{self.target_label}'")
            
            # Feed the data to the K-NN model
            self.classifier.register_tap(features, self.target_label)
            
            # Automatically revert to normal listening when done
            if self.taps_collected >= self.taps_needed:
                print(f"[+] Successfully learned '{self.target_label}'. Returning to execute mode.")
                self.mode = "execute"
                self.target_label = None
                
        elif self.mode == "execute":
            # --- ROUTE B: Normal operation ---
            prediction = self.classifier.predict(features)
            
            if prediction != "Unknown":
                print(f"[!] Action Triggered: Detected {tap_count} tap(s) on {prediction}")
                # TODO: Trigger the corresponding Fauxmo virtual switch here