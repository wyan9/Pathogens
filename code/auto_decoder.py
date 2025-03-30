import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code), code


class FeatureSensitivityAnalyzer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X):

        X_scaled = self.scaler.fit_transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32)
    
    def train_model(self, X, epochs=200, batch_size=64):

        dataset = TensorDataset(self.preprocess_data(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                
                recon_x, _ = self.model(x)
                loss = criterion(recon_x, x)

                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loss_history.append(total_loss/len(loader))
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")

        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Convergence')
        plt.show()
    
    def calculate_sensitivity(self, X, noise_scale=0.2):

        X_tensor = self.preprocess_data(X).to(self.device)
        base_recon = self.model(X_tensor)[0].detach()
        base_error = torch.mean((base_recon - X_tensor)**2, dim=0)
        
        sensitivity_scores = []
        for i in range(X.shape[1]):

            X_perturbed = X_tensor.clone()
            perturbation = torch.randn(X_perturbed.size(0)) * noise_scale * torch.std(X_perturbed[:,i])
            X_perturbed[:,i] += perturbation.to(self.device)
            

            recon_perturbed = self.model(X_perturbed)[0].detach()
            perturbed_error = torch.mean((recon_perturbed - X_tensor)**2, dim=0)

            delta_error = torch.mean(perturbed_error - base_error)
            sensitivity = delta_error / (perturbation.std().item() + 1e-8)
            sensitivity_scores.append(sensitivity.item())
        
        return np.array(sensitivity_scores)
    
    def visualize_importance(self, scores, feature_names):

        indices = np.argsort(scores)[::-1]  
        plt.figure(figsize=(10,6))
        

        plt.barh(range(len(indices)), scores[indices], align='center') 
        
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Sensitivity Score')
        plt.title('Feature Importance via Reconstruction Sensitivity')
        plt.gca().invert_yaxis()  
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# 示例使用
if __name__ == "__main__":

    np.random.seed(42)
    df = pd.read_excel('temp.xlsx', sheet_name='Sheet1')
    variances = df.var()
    print(variances)
    X = df[['Survival capability', 'Pathogenicity', 'Public health relevance']].values
    feature_names = [f"Feature_{i+1}" for i in range(3)]
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Autoencoder(input_dim=3, latent_dim=100)
    analyzer = FeatureSensitivityAnalyzer(model, device=device)
    

    analyzer.train_model(X, epochs=100)

    sensitivity = analyzer.calculate_sensitivity(X, noise_scale=0.3)

    analyzer.visualize_importance(sensitivity, feature_names)