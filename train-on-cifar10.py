# CREDIT: https://www.youtube.com/watch?v=S_il77Ttrmg&list=WL&index=4

import torchvision
import torch
import numpy as np
import wandb
from tqdm import tqdm 

from model import UNet, DiffusionModel
from utils import transform, plot_spatial_noise_distribution, plot_noise_distribution

wandb.login()

device = torch.device("cuda:0")

BATCH_SIZE = 256
NO_EPOCHS = 160
PRINT_FREQUENCY = 20
LR = 0.001
VERBOSE = True

unet = UNet(labels=True)
unet.to(device)

diffusion_model = DiffusionModel()

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

wandb.init(
  project='train-diffusion-model-on-cifar10',
  # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10):
  # name=f"XXX",
  config={
  "learning_rate": LR,
  "architecture": "diffusion",
  "dataset": "cifar10",
  "epochs": NO_EPOCHS,
  })

for epoch in tqdm(range(NO_EPOCHS)):
    mean_epoch_loss = []
    mean_epoch_loss_val = []
    
    for batch, label in trainloader:
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)
        batch = batch.to(device)
        batch_noisy, noise = diffusion_model.forward(batch, t, device) 
        predicted_noise = unet(batch_noisy, t, labels = label.reshape(-1,1).float().to(device))

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(noise, predicted_noise) 
        mean_epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    for batch, label in testloader:
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)
        batch = batch.to(device)

        batch_noisy, noise = diffusion_model.forward(batch, t, device) 
        predicted_noise = unet(batch_noisy, t, labels = label.reshape(-1,1).float().to(device))

        loss = torch.nn.functional.mse_loss(noise, predicted_noise) 
        mean_epoch_loss_val.append(loss.item())

    wandb.log({
        "train loss": np.mean(mean_epoch_loss),
        "validation loss": np.mean(mean_epoch_loss_val)
    })

    if epoch % PRINT_FREQUENCY == 0:
        print('---')
        print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")
        if VERBOSE:
            with torch.no_grad():
                plot_spatial_noise_distribution(noise[0], predicted_noise[0])
                plot_noise_distribution(noise, predicted_noise)
                
        torch.save(unet.state_dict(), f"epoch: {epoch}")
        
wandb.finish()
