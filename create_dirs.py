import os

base_path = r"e:\emotion-recognition\data"

emotions = ["angry", "happy", "sad", "surprise", "neutral", "fear", "disgust"]
splits = ["train", "test"]

for split in splits:
    for emotion in emotions:
        dir_path = os.path.join(base_path, split, emotion)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")

print("\nDirectory structure created successfully!")
