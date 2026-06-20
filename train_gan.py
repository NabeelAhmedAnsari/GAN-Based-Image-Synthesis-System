import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from generator import Generator
from discriminator import Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64
noise_dim = 100
epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(epochs):

    for imgs, _ in loader:

        imgs = imgs.to(device)

        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # Train Discriminator
        noise = torch.randn(imgs.size(0), noise_dim).to(device)
        fake_imgs = G(noise)

        real_loss = criterion(D(imgs), real)
        fake_loss = criterion(D(fake_imgs.detach()), fake)

        d_loss = real_loss + fake_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        g_loss = criterion(D(fake_imgs), real)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch} Loss D:{d_loss.item()} Loss G:{g_loss.item()}")

torch.save(G.state_dict(), "model/generator.pth")