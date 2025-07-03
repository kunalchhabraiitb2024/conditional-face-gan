import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

# ...existing code...

class FaceDataset(Dataset):
    """Custom dataset for faces extracted from WIDER FACE"""
    def __init__(self, data_path='./data/faces', img_size=128):
        self.data_path = data_path
        self.img_size = img_size
        
        # Get all image files
        import os
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            self.image_files.extend(glob.glob(os.path.join(data_path, ext)))
        
        print(f"Found {len(self.image_files)} face images in {data_path}")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image

def get_dataloader(data_path='./data/faces', batch_size=32, img_size=128, num_workers=4):
    """Get dataloader for custom face dataset"""
    dataset = FaceDataset(data_path=data_path, img_size=img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def denormalize_image(tensor):
    """Denormalize image tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1.0) / 2.0

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    tensor = denormalize_image(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Take first image
    
    # Convert to numpy and transpose
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)