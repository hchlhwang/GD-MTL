import argparse


def parseTag():
  parser = argparse.ArgumentParser(description='SegNet, single-task')
  parser.add_argument('--task', default='semantic', type=str, help="Task: semantic depth, normal")
  parser.add_argument('--data', default='preprocessed', type=str, help="Data path")
  option = parser.parse_args()
