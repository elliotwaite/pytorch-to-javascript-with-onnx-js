import torch
import torchvision


def main():
  train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST(
          'data', train=True, download=True,
          transform=torchvision.transforms.Compose([

              # torchvision.transforms.RandomAffine(
              #     degrees=30),

              # torchvision.transforms.RandomAffine(
              #     degrees=0, translate=(0.0, 0.5)),

              # torchvision.transforms.RandomAffine(
              #     degrees=0, translate=(0.5, 0.5)),

              # torchvision.transforms.RandomAffine(
              #     degrees=0, scale=(0.25, 1)),

              # torchvision.transforms.RandomAffine(
              #     degrees=0, shear=(-30, 30, -30, 30)),

              torchvision.transforms.RandomAffine(
                  degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
                  shear=(-30, 30, -30, 30)),

              torchvision.transforms.ToTensor(),
          ])),
      batch_size=800)
  inputs_batch, labels_batch = next(iter(train_loader))
  grid = torchvision.utils.make_grid(inputs_batch, nrow=40, pad_value=1)
  torchvision.utils.save_image(grid, 'inputs_batch_preview.png')


if __name__ == '__main__':
  main()
