import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import seaborn as sns

class GANMetrics:
    """Collection of metrics for evaluating GAN performance"""
    
    @staticmethod
    def frechet_inception_distance(real_features, fake_features):
        """
        Calculate Fréchet Inception Distance (FID)
        Lower is better
        """
        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return float(fid)
    
    @staticmethod
    def embedding_consistency_score(real_embeddings, fake_embeddings):
        """
        Calculate how consistent generated images are with their conditioning embeddings
        Higher is better (max = 1.0)
        """
        similarities = []
        for i in range(len(real_embeddings)):
            sim = cosine_similarity(
                real_embeddings[i:i+1], 
                fake_embeddings[i:i+1]
            )[0, 0]
            similarities.append(sim)
        
        return np.mean(similarities), np.std(similarities)
    
    @staticmethod
    def discriminator_score_analysis(d_scores_real, d_scores_fake):
        """
        Analyze discriminator scores
        Good generator: fake scores should be close to 0.5
        Good discriminator: real scores close to 1, fake scores close to 0
        """
        return {
            'real_mean': np.mean(d_scores_real),
            'real_std': np.std(d_scores_real),
            'fake_mean': np.mean(d_scores_fake),
            'fake_std': np.std(d_scores_fake),
            'score_separation': np.mean(d_scores_real) - np.mean(d_scores_fake)
        }
    
    @staticmethod
    def diversity_score(embeddings, threshold=0.9):
        """
        Calculate diversity of generated embeddings
        Higher is better - indicates diverse generation
        """
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Remove diagonal (self-similarity)
        mask = np.eye(len(similarities), dtype=bool)
        similarities = similarities[~mask]
        
        # Calculate diversity metrics
        mean_similarity = np.mean(similarities)
        diversity_score = 1 - mean_similarity
        
        # Count highly similar pairs
        high_similarity_pairs = np.sum(similarities > threshold)
        total_pairs = len(similarities)
        
        return {
            'diversity_score': diversity_score,
            'mean_similarity': mean_similarity,
            'high_similarity_ratio': high_similarity_pairs / total_pairs
        }
    
    @staticmethod
    def plot_score_distributions(d_scores_real, d_scores_fake, save_path=None):
        """Plot discriminator score distributions"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(d_scores_real, bins=30, alpha=0.7, label='Real Images', color='blue')
        plt.hist(d_scores_fake, bins=30, alpha=0.7, label='Fake Images', color='red')
        
        plt.xlabel('Discriminator Score')
        plt.ylabel('Frequency')
        plt.title('Discriminator Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_embedding_similarity_heatmap(embeddings, save_path=None):
        """Plot embedding similarity heatmap"""
        similarities = cosine_similarity(embeddings)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarities, cmap='viridis', center=0, 
                   square=True, cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Embedding Similarity Matrix')
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Index')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

class TrainingMonitor:
    """Monitor training progress and detect issues"""
    
    def __init__(self):
        self.losses_g = []
        self.losses_d = []
        self.d_real_scores = []
        self.d_fake_scores = []
    
    def update(self, loss_g, loss_d, d_real, d_fake):
        """Update metrics"""
        self.losses_g.append(loss_g)
        self.losses_d.append(loss_d)
        self.d_real_scores.append(d_real)
        self.d_fake_scores.append(d_fake)
    
    def detect_mode_collapse(self, window=10, threshold=0.01):
        """Detect potential mode collapse"""
        if len(self.losses_g) < window:
            return False
        
        recent_g_losses = self.losses_g[-window:]
        variance = np.var(recent_g_losses)
        
        return variance < threshold
    
    def detect_training_instability(self, window=10, threshold=2.0):
        """Detect training instability"""
        if len(self.losses_g) < window:
            return False
        
        recent_losses = self.losses_g[-window:]
        mean_loss = np.mean(recent_losses)
        max_deviation = max(abs(loss - mean_loss) for loss in recent_losses)
        
        return max_deviation > threshold * mean_loss
    
    def get_training_health(self):
        """Get overall training health assessment"""
        if len(self.losses_g) < 5:
            return "Insufficient data"
        
        issues = []
        
        # Check for mode collapse
        if self.detect_mode_collapse():
            issues.append("Potential mode collapse detected")
        
        # Check for instability
        if self.detect_training_instability():
            issues.append("Training instability detected")
        
        # Check discriminator balance
        recent_d_real = np.mean(self.d_real_scores[-10:]) if len(self.d_real_scores) >= 10 else 0
        recent_d_fake = np.mean(self.d_fake_scores[-10:]) if len(self.d_fake_scores) >= 10 else 0
        
        if recent_d_real < 0.6:
            issues.append("Discriminator too weak on real images")
        elif recent_d_real > 0.95:
            issues.append("Discriminator too strong on real images")
        
        if recent_d_fake > 0.4:
            issues.append("Generator fooling discriminator too easily")
        elif recent_d_fake < 0.05:
            issues.append("Generator not fooling discriminator enough")
        
        if not issues:
            return "Training appears healthy"
        else:
            return "; ".join(issues)
    
    def plot_training_progress(self, save_path=None):
        """Plot comprehensive training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.losses_g, label='Generator', color='blue')
        axes[0, 0].plot(self.losses_d, label='Discriminator', color='red')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator scores
        axes[0, 1].plot(self.d_real_scores, label='Real Images', color='green')
        axes[0, 1].plot(self.d_fake_scores, label='Fake Images', color='orange')
        axes[0, 1].set_title('Discriminator Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss ratio
        if len(self.losses_d) > 0:
            loss_ratios = [g/d if d > 0 else 0 for g, d in zip(self.losses_g, self.losses_d)]
            axes[1, 0].plot(loss_ratios, color='purple')
            axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('G_loss / D_loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Score difference
        if len(self.d_real_scores) > 0 and len(self.d_fake_scores) > 0:
            score_diffs = [r - f for r, f in zip(self.d_real_scores, self.d_fake_scores)]
            axes[1, 1].plot(score_diffs, color='brown')
            axes[1, 1].set_title('Discriminator Score Difference (Real - Fake)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score Difference')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()