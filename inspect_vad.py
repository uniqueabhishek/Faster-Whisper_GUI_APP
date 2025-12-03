import faster_whisper.vad
import inspect

print("Attributes of faster_whisper.vad:")
for name in dir(faster_whisper.vad):
    if not name.startswith("__"):
        print(name)

print("\nSileroVADModel init signature:")
print(inspect.signature(faster_whisper.vad.SileroVADModel.__init__))

print("\nChecking if we can instantiate SileroVADModel with a path:")
try:
    # Try to instantiate with our local path
    import os
    local_path = os.path.abspath("assets/silero_vad.onnx")
    print(f"Local path: {local_path}")
    model = faster_whisper.vad.SileroVADModel(local_path)
    print("Successfully instantiated SileroVADModel with local path")
except Exception as e:
    print(f"Failed to instantiate: {e}")
