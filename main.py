from src.audio.listener import AudioListener
from src.audio.new_processor import SignalProcessor
from src.ml.classifier import TapClassifier
from src.controller import SwitchController
import pyaudio
import threading
import sys

def list_input_devices():
    p = pyaudio.PyAudio()
    print("--- Available Audio Input Devices ---\n")
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        
        # We only care about devices with input channels (microphones)
        if device_info.get('maxInputChannels') > 0:
            name = device_info.get('name')
            channels = device_info.get('maxInputChannels')
            sample_rate = int(device_info.get('defaultSampleRate'))
            
            print(f"Index [{i}] : {name}")
            print(f"           - Max Channels: {channels}")
            print(f"           - Default Rate: {sample_rate} Hz\n")
            
    p.terminate()


def main(model, device_index):
    # 1. Initialize the core components
    processor = SignalProcessor(model = model)
    classifier = TapClassifier(n_neighbors=3, distance_threshold=20.0)
    
    # Assuming your USB mic is device_index 2 (from our previous test)
    listener = AudioListener(processor, device_index=device_index)
    controller = SwitchController(classifier)

    # 2. Spin up the Audio Listener in a background Daemon Thread
    audio_thread = threading.Thread(
        target=listener .listen, 
        args=(controller.audio_callback,), 
        daemon=True
    )
    audio_thread.start()

    # 3. The Main Thread CLI Loop
    print("\n--- Universal Switch Command Line ---")
    print("Commands:")
    print("  register <label> : Start teaching a new tap sound")
    print("  quit             : Exit the application")
    print("-------------------------------------\n")

    try:
        while True:
            # The main thread blocks here waiting for your keyboard input
            user_input = input(">> ").strip().lower()
            
            if user_input == "quit":
                print("[*] Shutting down...")
                sys.exit(0)
                
            elif user_input.startswith("register "):
                # Parse the label from the command (e.g., "register nightstand")
                label = user_input.split(" ", 1)[1]
                controller.set_registration_mode(label=label, taps=10)
                
            else:
                print("[!] Unknown command. Try 'register <label>' or 'quit'.")
                
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    list_input_devices()

    main(model = "pann", device_index=0)
    # processor = SignalProcessor(model="pann")
    # listener = AudioListener(processor=processor,device_index=1)
    # classifier = TapClassifier(distance_threshold=45.0)
    # controller = SwitchController(classifier=classifier)

    # listener.test_file("tests/file 1.m4a")
    
    # controller.set_registration_mode("Wooden table", 10)
    # listener.listen(controller.audio_callback)