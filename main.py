import torch
from model import SegNet
from util import parseTag


def main():

  parseTag()

  # Set GPU
  device = torch.cuda.device("cuda")

  # Send model to GPU
  model = SegNet().to(device)


if __name__ == '__main__':
  main()
