import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# Make wandb optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Training will continue without logging to wandb.")

from model import FaceEncoder, Generator, Discriminator, weights_init
from dataset import get_dataloader, denormalize_image

class FaceGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.encoder = FaceEncoder(
            embedding_dim=config['embedding_dim'],
            freeze_backbone=config['freeze_encoder']
        ).to(self.device)
        
        self.generator = Generator(
            embedding_dim=config['embedding_dim'],
            noise_dim=config['noise_dim']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            embedding_dim=config['embedding_dim']
        ).to(self.device)
        
        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.opt_g = optim.Adam(
            list(self.generator.parameters()) + 
            ([] if config['freeze_encoder'] else list(self.encoder.parameters())),
            lr=config['lr_g'], 
            betas=(config['beta1'], 0.999)
        )
        
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_d'],
            betas=(config['beta1'], 0.999)
        )
        
        # Data loader
        self.dataloader = get_dataloader(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers']
        )
        
        # Fixed noise for visualization
        self.fixed_noise = torch.randn(16, config['noise_dim'], device=self.device)
        
        # Metrics tracking
        self.losses_g = []
        self.losses_d = []
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
    def log_metrics(self, metrics):
        """Log metrics to wandb if available, otherwise print"""
        if wandb.run is not None:
            wandb.log(metrics)
        else:
            if 'epoch_loss_g' in metrics:
                print(f"Epoch {metrics.get('epoch', 0)}: G_loss={metrics['epoch_loss_g']:.4f}, D_loss={metrics.get('epoch_loss_d', 0):.4f}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        self.encoder.train()
        
        epoch_loss_g = 0
        epoch_loss_d = 0
        
        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for i, real_images in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Labels
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)
            
            # Get embeddings from real images
            with torch.no_grad() if self.config['freeze_encoder'] else torch.enable_grad():
                real_embeddings = self.encoder(real_images)
            
            # ============ Train Discriminator ============
            self.opt_d.zero_grad()
            
            # Real images
            output_real = self.discriminator(real_images, real_embeddings)
            loss_d_real = self.criterion(output_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.config['noise_dim'], device=self.device)
            fake_images = self.generator(real_embeddings, noise)
            output_fake = self.discriminator(fake_images.detach(), real_embeddings)
            loss_d_fake = self.criterion(output_fake, fake_labels)
            
            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.opt_d.step()
            
            # ============ Train Generator ============
            self.opt_g.zero_grad()
            
            # Generate fake images and get discriminator output
            output_fake = self.discriminator(fake_images, real_embeddings)
            loss_g = self.criterion(output_fake, real_labels)
            
            # Add reconstruction loss for better conditioning
            if self.config['use_reconstruction_loss']:
                recon_loss = nn.MSELoss()(fake_images, real_images)
                loss_g += self.config['recon_weight'] * recon_loss
            
            loss_g.backward()
            self.opt_g.step()
            
            # Update metrics
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_g.item():.4f}',
                'D_loss': f'{loss_d.item():.4f}',
                'D_real': f'{output_real.mean().item():.3f}',
                'D_fake': f'{output_fake.mean().item():.3f}'
            })
            
            # Log to wandb
            if i % self.config['log_interval'] == 0:
                self.log_metrics({
                    'batch_loss_g': loss_g.item(),
                    'batch_loss_d': loss_d.item(),
                    'D_real_score': output_real.mean().item(),
                    'D_fake_score': output_fake.mean().item(),
                    'epoch': epoch,
                    'batch': i
                })
        
        # Average losses for the epoch
        avg_loss_g = epoch_loss_g / len(self.dataloader)
        avg_loss_d = epoch_loss_d / len(self.dataloader)
        
        self.losses_g.append(avg_loss_g)
        self.losses_d.append(avg_loss_d)
        
        return avg_loss_g, avg_loss_d
    
    def generate_samples(self, epoch):
        """Generate sample images for visualization"""
        self.generator.eval()
        self.encoder.eval()
        
        with torch.no_grad():
            # Get some real images for reference
            real_batch = next(iter(self.dataloader))[:16].to(self.device)
            real_embeddings = self.encoder(real_batch)
            
            # Generate fake images
            fake_images = self.generator(real_embeddings, self.fixed_noise)
            
            # Create comparison grid
            comparison = torch.cat([real_batch, fake_images], dim=0)
            grid = make_grid(denormalize_image(comparison), nrow=8, padding=2)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    f'verification_samples_epoch_{epoch}': wandb.Image(grid.permute(1, 2, 0).cpu().numpy()),
                    'epoch': epoch
                })
            
            # Save locally
            plt.figure(figsize=(12, 6))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.title(f'VERIFICATION - Epoch {epoch+1}: Top row = Real, Bottom row = Generated')
            plt.tight_layout()
            plt.savefig(f'results/verification_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_g_state_dict': self.opt_g.state_dict(),
            'optimizer_d_state_dict': self.opt_d.state_dict(),
            'losses_g': self.losses_g,
            'losses_d': self.losses_d,
            'config': self.config
        }
        
        filename = f'results/verification_checkpoint_epoch_{epoch+1}.pth'
        if is_best:
            filename = 'results/verification_best_model.pth'
            
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def train(self, start_epoch=0):
        """Main training loop"""
        print("Starting training from epoch", start_epoch)
        best_loss_g = float('inf')
        
        for epoch in range(start_epoch, self.config['epochs']):
            # Train for one epoch
            avg_loss_g, avg_loss_d = self.train_epoch(epoch)
            
            # Log epoch metrics
            self.log_metrics({
                'epoch_loss_g': avg_loss_g,
                'epoch_loss_d': avg_loss_d,
                'epoch': epoch
            })
            
            print(f'Epoch [{epoch+1}/{self.config["epochs"]}] - G_loss: {avg_loss_g:.4f}, D_loss: {avg_loss_d:.4f}')
            
            # Generate samples every 2 epochs for quick verification
            if (epoch + 1) % 2 == 0:
                self.generate_samples(epoch)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                
                # Save best model
                if avg_loss_g < best_loss_g:
                    best_loss_g = avg_loss_g
                    self.save_checkpoint(epoch, is_best=True)
        
        print("VERIFICATION training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./face_dataset', help='Path to face dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--test_run', action='store_true', help='Run quick test with 10 epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    
    # Configuration
    config = {
        # Model parameters
        'embedding_dim': 512,
        'noise_dim': 100,
        'img_size': 128,
        'freeze_encoder': True,
        
        # Training parameters
        'batch_size': args.batch_size,
        'epochs': 10 if args.test_run else args.epochs,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'beta1': 0.5,
        'num_workers': 4,
        
        # Loss parameters
        'use_reconstruction_loss': True,
        'recon_weight': 10.0,
        
        # Logging and saving
        'log_interval': 25,
        'sample_interval': 2,
        'checkpoint_interval': 5,
        
        # Dataset
        'data_path': args.data_path,
    }
    
    # Initialize wandb for verification with public visibility
    run_name = f"{'test' if args.test_run else 'full'}-run-{args.epochs}epochs"
    wandb.init(
            project="face-generation-verification",
            config=config,
            name=run_name,
            mode="online",
            entity=None  # Make run public
    )
    
    # Create trainer and start training
    trainer = FaceGANTrainer(config)
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        
        # Load model states
        trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
        trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Load optimizer states
        trainer.opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        trainer.opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load losses history
        trainer.losses_g = checkpoint['losses_g']
        trainer.losses_d = checkpoint['losses_d']
        
        # Set starting epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    trainer.train(start_epoch=start_epoch)
    
    # Finish wandb run if used
    if wandb.run is not None:
        wandb.finish()
    
    print("\n🎯 Training COMPLETE!")
    print("Check results/ folder for generated samples and checkpoints")