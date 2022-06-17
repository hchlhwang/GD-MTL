import torch
from model import SegNet
from util import parseTag


def main():

  parseTag()

  # Set GPU
  device = torch.cuda.device('cuda:0' if torch.cuda.is_available else 'cpu')

  # Send model to GPU
  model = SegNet().to(device)


if __name__ == '__main__':
  main()
