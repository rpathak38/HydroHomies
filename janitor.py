import os
from PIL import Image


def clean_dataset(folder):
    num_removed = 0
    for root, _, files in os.walk(folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it's a valid image
            except (IOError, OSError, SyntaxError) as e:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                num_removed += 1
    print(f"Total corrupted images removed: {num_removed}")


# Clean train, validation, and test datasets
clean_dataset('./data/train')
clean_dataset('./data/valid')
clean_dataset('./data/test')
