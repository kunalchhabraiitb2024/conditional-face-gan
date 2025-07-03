import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class FaceEncoder(nn.Module):
    """Pre-trained face encoder using FaceNet"""
    def __init__(self, embedding_dim=512, freeze_backbone=True):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2')
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add projection layer to get desired embedding dimension
        self.projection = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # Resize to 160x160 for FaceNet
        x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

class Generator(nn.Module):
    """Improved Generator with better architecture"""
    def __init__(self, embedding_dim=512, noise_dim=100, img_channels=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        
        # Better embedding processing
        self.embed_proj = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True)
        )
        
        # Noise projection
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True)
        )
        
        # Combined input
        input_dim = 1024
        
        # Improved generator with residual connections
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, embeddings, noise):
        # Better feature combination
        embed_proj = self.embed_proj(embeddings)
        noise_proj = self.noise_proj(noise)
        
        combined = torch.cat([embed_proj, noise_proj], dim=1)
        combined = combined.view(combined.size(0), -1, 1, 1)
        
        # Progressive generation
        x = self.initial(combined)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.final(x)
        
        return output

class Discriminator(nn.Module):
    """Conditional Discriminator that takes images and embeddings"""
    def __init__(self, embedding_dim=512, img_channels=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Image processing path
        self.img_conv = nn.Sequential(
            # (3, 128, 128) -> (64, 64, 64)
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 64, 64) -> (128, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 32, 32) -> (256, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, 16, 16) -> (512, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (512, 8, 8) -> (1024, 4, 4)
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Embedding processing
        self.embed_proj = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024 * 4 * 4)
        )
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 1, 4, 1, 0, bias=False),  # 1024 + 1024 = 2048
            nn.Sigmoid()
        )
        
    def forward(self, images, embeddings):
        # Process image
        img_features = self.img_conv(images)  # (batch, 1024, 4, 4)
        
        # Process embeddings
        embed_features = self.embed_proj(embeddings)  # (batch, 1024*4*4)
        embed_features = embed_features.view(-1, 1024, 4, 4)  # (batch, 1024, 4, 4)
        
        # Concatenate features
        combined = torch.cat([img_features, embed_features], dim=1)  # (batch, 2048, 4, 4)
        
        # Final classification
        output = self.classifier(combined)
        return output.view(-1, 1).squeeze(1)

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)