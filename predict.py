-"""
Melody Diffuser - Replicate Predictor
=====================================
Wraps the diffusion model for Replicate API deployment.
"""

from cog import BasePredictor, Input
import torch
import numpy as np
from typing import List
import os
import urllib.request

# Import your model class
# Make sure model.py is in the same folder as this script
from model import MelodyDiffusor 

# --- Helper Functions ---
def get_betas(start, end, steps):
    return torch.linspace(start, end, steps)

def add_noise(x, noise_prob, vocab_size):
    """Add discrete noise by randomly replacing tokens."""
    # Ensure noise_prob matches x dimensions
    noise_prob = noise_prob.expand_as(x.float())
    mask = torch.rand_like(x.float()) < noise_prob
    random_tokens = torch.randint_like(x, 0, vocab_size)
    return torch.where(mask, random_tokens, x)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Constants ---
        self.seq_len = 64
        self.vocab_size = 130
        self.diffusion_steps = 64
        self.model = MelodyDiffusor(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            dim=512,            # <--- CHECK THIS (Likely 128, 256, or 512)
            n_layers=6,         # <--- CHECK THIS (Likely 4, 6, or 8)
            n_heads=8,          # <--- CHECK THIS (Likely 4 or 8)
            ffn_inner_dim=2048  # <--- CHECK THIS (Usually 4x dim, e.g. 2048 or 1024)
        )

        # --- 2. Download Weights ---
        weights_url = "https://huggingface.co/DuncanLarz/Melody-Diffuser/resolve/main/Best-Melody-Diffuser.pth"
        local_weights_path = "./model.pth"
        
        if not os.path.exists(local_weights_path):
            print(f"Downloading weights from {weights_url}...")
            urllib.request.urlretrieve(weights_url, local_weights_path)

        # --- 3. Load Weights into Model ---
        state_dict = torch.load(local_weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Move to GPU and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")

        # --- 4. Precompute Diffusion Schedule ---
        betas = get_betas(1e-4, 0.02, self.diffusion_steps).to(self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alpha_cum = alphas_cumprod.to(self.device)

    def predict(
        self,
        gestures: str = Input(
            description="Comma-separated gesture tokens (64 integers, each 0-7). Example: '0,2,3,4...'",
            default="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
        ),
        temperature: float = Input(
            description="Sampling temperature (higher = more random)",
            default=1.0, ge=0.1, le=2.0
        ),
        top_p: float = Input(
            description="Nucleus sampling threshold",
            default=0.95, ge=0.1, le=1.0
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale (higher = more adherence to gestures)",
            default=2.0, ge=1.0, le=5.0
        ),
        use_cfg: bool = Input(
            description="Whether to use classifier-free guidance",
            default=True
        )
    ) -> str:
        """
        Generate a melody from gesture conditioning.
        Returns: Comma-separated MIDI tokens (64 integers).
        """
        # Parse gesture input
        try:
            gesture_list = [int(x.strip()) for x in gestures.split(",")]
        except ValueError:
            raise ValueError("Gestures must be comma-separated integers (0-7)")
        
        # Validate input length
        if len(gesture_list) != 64:
            raise ValueError(f"Expected 64 gesture tokens, got {len(gesture_list)}")
        
        if not all(0 <= g <= 7 for g in gesture_list):
            raise ValueError("All gesture tokens must be between 0-7")
        
        # Convert to tensor
        cond_tensor = torch.tensor(gesture_list, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Run Sampling
        melody = self._sample(
            cond_tensor,
            temperature=temperature,
            top_p=top_p,
            cfg_scale=cfg_scale,
            use_cfg=use_cfg
        )
        
        # Return as string
        return ",".join(str(int(x)) for x in melody)
    
    def _sample(self, cond, temperature, top_p, cfg_scale, use_cfg):
        """Run diffusion sampling."""
        B, L = 1, self.seq_len
        x = torch.randint(0, self.vocab_size, (B, L), device=self.device)
        
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                cond_logits = self.model(x, t_tensor, cond)
                
                if use_cfg and cfg_scale != 1.0:
                    uncond_logits = self.model(x, t_tensor, None)
                    logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                else:
                    logits = cond_logits
            
            # Temperature
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0
            
            # Avoid division by zero if all probs are masked (rare)
            sum_probs = sorted_probs.sum(dim=-1, keepdim=True)
            sorted_probs = sorted_probs / (sum_probs + 1e-8)
            
            # Restore order and sample
            orig_probs = torch.zeros_like(sorted_probs)
            orig_probs.scatter_(-1, sorted_indices, sorted_probs)
            x_0_pred = torch.multinomial(orig_probs.view(-1, self.vocab_size), 1).view(B, L)
            
            if t == 0:
                x = x_0_pred
                break
            
            # Add noise for next step using precomputed alpha_cum
            noise_prob = (1 - self.alpha_cum[t-1]).view(-1, 1)
            x = add_noise(x_0_pred, noise_prob, self.vocab_size)
        
        return x.squeeze().cpu().numpy()
