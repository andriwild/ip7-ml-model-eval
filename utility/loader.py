import os

def load_all_images(folder_path: str, num_images: int) -> list[str]:
    images = []
    img_ext: list[str] = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in img_ext):
            img_path: str = os.path.join(folder_path, file)
            images.append(img_path)
            if len(images) >= num_images:
                break

    return images

