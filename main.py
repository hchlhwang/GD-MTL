import torch
import torch.optim as optim
from model import SegNet
from util import parseTag


def main():

  parseTag()

  # Set GPU in desktop or mac m1
  if torch.cuda.is_available():
    device = torch.cuda.device('cuda:0' if torch.cuda.is_available else 'cpu')
  else:
    device = torch.device("mps")

  print(f"Using GPS device: {device}")

  # Send model to GPU
  model = SegNet().to(device)

  # Optimze params
  optimizer = optim.Adam(SegNet.parameters(), lr=1e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

  print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet),
                                                           count_parameters(SegNet) / 24981069))
  print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')


  # define dataset
  dataset_path = opt.dataroot
  if opt.apply_augmentation:
      nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
      print('Applying data augmentation on NYUv2.')
  else:
      nyuv2_train_set = NYUv2(root=dataset_path, train=True)
      print('Standard training strategy without data augmentation.')

  nyuv2_test_set = NYUv2(root=dataset_path, train=False)

  batch_size = 2
  nyuv2_train_loader = torch.utils.data.DataLoader(
      dataset=nyuv2_train_set,
      batch_size=batch_size,
      shuffle=True)

  nyuv2_test_loader = torch.utils.data.DataLoader(
      dataset=nyuv2_test_set,
      batch_size=batch_size,
      shuffle=False)

  # Train and evaluate single-task network
  single_task_trainer(nyuv2_train_loader,
                      nyuv2_test_loader,
                      SegNet,
                      device,
                      optimizer,
                      scheduler,
                      opt,
                      200)


if __name__ == '__main__':
  main()
