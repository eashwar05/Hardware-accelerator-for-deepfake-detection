import os
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, spatial_transform, freq_transform):
        self.spatial_transform, self.freq_transform = spatial_transform, freq_transform
        self.image_paths, self.labels = [], []
        for label, sub_dir in enumerate(['real', 'fake']):
            dir_path = os.path.join(root_dir, sub_dir)
            for fname in os.listdir(dir_path):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(dir_path, fname))
                    self.labels.append(label) # 0 for 'real', 1 for 'fake'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        spatial_tensor = self.spatial_transform(image)
        freq_tensor = self.freq_transform(image)
        return spatial_tensor, freq_tensor, label