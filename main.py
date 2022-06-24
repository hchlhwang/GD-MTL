import torch
import torch.optim as optim
from model import *
from util import count_parameters, model_fit, single_task_trainer
from dataset import NYUV2
from torch.utils.data import DataLoader


def main():

    # Set path
    dataRoot = '/home/hochul/Repository/mtan/im2im_pred/preprocessed'

    # Set GPU in desktop or mac m1
    if torch.cuda.is_available():
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        print(f'Using {torch.cuda.get_device_name(0)} GPS device')
    else:
        device = torch.device('mps')
        print('Using mac m1')

    # Set hyperparameters
    batchSize = 64
    learningRate = 1e-4
    epochs = 200

    # Define dataset
    nyuv2TrainData = NYUV2(dataRoot)
    nyuv2TestData = NYUV2(dataRoot, train=False)

    # Define dataloader
    nyuv2TrainLoader = DataLoader(dataset=nyuv2TrainData,
                                  batch_size=batchSize,
                                  shuffle=True,
                                  )
    nyuv2TrainLoader = DataLoader(dataset=nyuv2TestData,
                                  batch_size=batchSize,
                                  shuffle=False,
                                  )

    # Set model
    model = SegNet().to(device)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # TODO
    # inference(loader,model,device,optimizer)




if __name__ == '__main__':
    main()
    # parser = argparse.ArgumentParser(description='SegNet, single-task')
    # parser.add_argument('--checkpoint', default='checkpoint', type=str, help="Checkpoint path")
    # parser.add_argument('--data', default='preprocessed', type=str, help="Data path")
    # parser.add_argument('-B','--batch-size', default='2', type=int, help="Batch size")
    # parser.add_argument('-L','--learning-rate', default='1e-3', type=float, help="Learning rate")
    # parser.add_argument('--task', default='semantic', type=str, help="Task: semantic depth, normal")
    # parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
    # parser.add_argument('--multitask', dest='multitask', action='store_true',
    #                     help='Multi-task (not single task)')
    # option = parser.parse_args()
