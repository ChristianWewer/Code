
import torch
import torch.nn as nn
import torch.nn.functional as F


class AEEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(AEEncoder, self).__init__()
        self.linear1 = nn.Linear(11, 5)
        self.linear2 = nn.Linear(5, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)



class AEDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(AEDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 5)
        self.linear2 = nn.Linear(5, 11)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        #z = nn.Linear(self.linear2(z))
        z = self.linear2(z)
        return z


class AE(nn.Module):
    def __init__(self, latent_dims):
        super(AE, self).__init__()
        self.encoder = AEEncoder(latent_dims)
        self.decoder = AEDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



# VAE

class VAEEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VAEEncoder, self).__init__()
        self.linear1 = nn.Linear(11, 10)
        self.linear2 = nn.Linear(10, 9)
        self.linear3 = nn.Linear(9, 8)
        self.linear4 = nn.Linear(8, 7)
        self.linear5 = nn.Linear(7,6)
        self.linear6 = nn.Linear(6,5)
        self.linear7 = nn.Linear(5, latent_dims)
        self.linear8 = nn.Linear(5, latent_dims)

        self.N = torch.distributions.Normal(0,1)
        self.kl = 0

        self.mu = 0
        self.sigma = 0
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        self.mu = self.linear7(x)
        self.sigma = torch.exp(self.linear8(x))
        z = self.mu + self.sigma*self.N.sample(self.mu.shape) # sample from the latent space
        self.kl = (self.sigma**2 + self.mu**2 - torch.log(self.sigma) - 1/2).sum() # KL divergence
        return z

    def draw_sample(self):
        return self.mu + self.sigma*self.N.sample(self.mu.shape)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(VAEDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 5)
        self.linear2 = nn.Linear(5, 6)
        self.linear3 = nn.Linear(6, 7)
        self.linear4 = nn.Linear(7, 8)
        self.linear5 = nn.Linear(8, 9)
        self.linear6 = nn.Linear(9, 10)
        self.linear7 = nn.Linear(10, 11)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = F.relu(self.linear5(z))
        z = F.relu(self.linear6(z))
        z = self.linear7(z)
        return z


class VAE(nn.Module):
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(latent_dims)
        self.decoder = VAEDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def draw_sample(self):
        return self.encoder.draw_sample()



class VAE_encoder_3hl(nn.Module):
    def __init__(self, latent_dims, layers):
        super(VAE_encoder_3hl, self).__init__()

        self.linear1 = nn.Linear(11, layers[0])
        self.linear2 = nn.Linear(layers[0], layers[1])
        self.linear3 = nn.Linear(layers[1], layers[2])
        self.linear4 = nn.Linear(layers[2], latent_dims)
        self.linear5 = nn.Linear(layers[2], latent_dims)
        

        self.N = torch.distributions.Normal(0,1)
        self.kl = 0

        self.mu = 0
        self.sigma = 0


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        self.mu = self.linear4(x)
        self.sigma = torch.exp(self.linear5(x))
        z = self.mu + self.sigma*self.N.sample(self.mu.shape) # sample from the latent space
        self.kl = (self.sigma**2 + self.mu**2 - torch.log(self.sigma) - 1/2).sum() # KL divergence
        return z

    def draw_sample(self):
        return self.mu + self.sigma*self.N.sample(self.mu.shape)

class VAE_decoder_3hl(nn.Module):
    def __init__(self, latent_dims, layers):
        super(VAE_decoder_3hl, self).__init__()
        self.linear1 = nn.Linear(latent_dims, layers[2])
        self.linear2 = nn.Linear(layers[2], layers[1])
        self.linear3 = nn.Linear(layers[1], layers[0])
        self.linear4 = nn.Linear(layers[0], 11)


    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = self.linear4(z)
        return z


class VAE_3hl(nn.Module):
    def __init__(self, latent_dims, layers):
        super(VAE_3hl, self).__init__()
        self.encoder = VAE_encoder_3hl(latent_dims, layers)
        self.decoder = VAE_decoder_3hl(latent_dims, layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def draw_sample(self):
        return self.encoder.draw_sample()