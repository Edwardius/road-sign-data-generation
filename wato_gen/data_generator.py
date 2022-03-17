import argparse
import os

def load_images_coco():

  return

def load_images_road_signs():
  
  return

def generate_data():
  
  return

def export():
  
  return


def generate_road_sign_data(args):
  ''' Generate synthetic dataset according to given args
  '''
  # Load images from coco

  # Load road sign data
  
  # Generate data

  # Export data

  return

def parse_args():
  '''Parse input arguments
  '''
  parser = argparse.ArgumentParser(description="Creates a dataset of road signs")
  # Path Parameters
  parser.add_argument("coco", default="/coco",
    help="The coco directory which contains the images and labels.")
  parser.add_argument("export", default="/wato_road_sign_data",
    help="The directory where images and labels will be created.")
  parser.add_argument("road_signs", default="/mnt/wato-drive/perception_2d/signs_yolo/just_signs",
    help="WATonomous' road signs data.")

  # Generation Parameters
  parser.add_argument("road_signs", default="/mnt/wato-drive/perception_2d/signs_yolo/just_signs",
    help="WATonomous' road signs data.")
  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = parse_args()

  generate_road_sign_data(args)
