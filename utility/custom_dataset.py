from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Callable
from pathlib import Path


@dataclass
class ImageSpecification:
    original: Image.Image
    resized: Image.Image
    horizontal_pad: int
    vertical_pad: int


# add padding to the image
# makes it square (to model dimensions) on the same aspect ratio
# (letterboxing)
def preprocessing(target_size: tuple[int,int], orig_image: Image.Image) -> ImageSpecification:
    
    iw, ih = orig_image.size
    w, h = target_size 
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    image_resized = orig_image.resize((nw, nh), Image.Resampling.BOX)
    assert image_resized.size[0] <= target_size[0]
    assert image_resized.size[1] <= target_size[1]

    # add gray padding
    new_image = Image.new("RGB", target_size, (128, 128, 128))     
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))

    padding_horizontal = new_image.size[0] - image_resized.size[0]
    padding_vertical   = new_image.size[1] - image_resized.size[1]

    return ImageSpecification(
            orig_image,
            new_image,
            padding_horizontal,
            padding_vertical,
            )


class FilesystemDataset(Dataset):
    def __init__( self, root: str, transform: Callable[[Image.Image], ImageSpecification]):

        self.extensions = ('.jpg', '.jpeg', '.png')
        self.img_dir = Path(root)
        self.img_files = [str(p) for p in self.img_dir.rglob('*') if p.is_file() and p.suffix.lower() in self.extensions]
        #self.img_files = [ f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) ]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx) -> ImageSpecification:
        try:
            image = Image.open(self.img_files[idx]).convert('RGB')
        except Exception as e:
            print(f"Error opening image {self.img_files[idx]}: {e}")
            return self.transform(Image.new('RGB', (480, 480)))
        return self.transform(image)


class InMemoryDateset(Dataset):
    def __init__(
            self, 
            images: list[Image.Image], 
            transform: Callable[[Image.Image], ImageSpecification]
            ):
        self.images= images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> ImageSpecification:
        return self.transform(self.images[idx])
