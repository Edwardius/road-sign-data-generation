import argparse
import glob  
import os
dirname = os.path.dirname(os.path.abspath(__file__))

from multiprocessing import Pool
       

def process_synth_data(data):


  try:
    pool.map(partial_func, params_list)
  except KeyboardInterrupt:
    print("....\nCaught KeyboardInterrupt, terminating workers")
    pool.terminate()
    pool.join()

  return

def generate_road_sign_data(args):
  ''' Generate synthetic dataset according to given args
      we assume that the number of augmented images is not bigger
      than the coco training dataset (~581000 images)
  '''
  # completing relative paths
  coco_dir = os.path.join(dirname, args.coco)
  road_sign_dir = os.path.join(dirname, args.road_signs)
  
  # iterator for coco images and their labels
  coco_image_iterator = glob.iglob(os.path.join(coco_dir, 'images', 'train2017', '*.jpg'))
  coco_label_iterator = glob.iglob(os.path.join(coco_dir, 'labels', 'train2017', '*.txt'))

  # what we will be sending into the pool for multiprocessing
  data_iterable = []

  # for each of the road sign categories
  for x in os.listdir(road_sign_dir):
    road_sign_class = x

    # iterator over the folder of augmented road_signs
    road_sign_iterator = glob.iglob(os.path.join(road_sign_dir, road_sign_class, '*.png'))
    finish_iteration = False

    # iterate over all the augmented images in the folder 
    while not finish_iteration:
      try:
        road_sign = next(road_sign_iterator)
      except StopIteration:
        finish_iteration = True
      else:
        data_iterable.append([road_sign, road_sign_class, 
          next(coco_image_iterator), next(coco_label_iterator)])
        
        if len(data_iterable) is args.batch_size:
          # send data_iterable in for processing
          process_synth_data(data_iterable)

          # empty data_iterable
          data_iterable.clear()
  
  data_iterable.clear()

  return

def parse_args():
  '''Parse input arguments
  '''
  parser = argparse.ArgumentParser(description="Creates a dataset of road signs")
  # Path Parameters
  parser.add_argument("coco", default="coco",
    help="The coco directory which contains the images and labels.")
  parser.add_argument("export", default="wato_road_sign_data",
    help="The directory where images and labels will be created.")
  parser.add_argument("road_signs", default="/mnt/wato-drive/perception_2d/signs_yolo/just_signs",
    help="WATonomous' road signs data.")

  # Generation Parameters
  parser.add_argument("batch_size", default=64,
    help="How many images do we want to process at a time?")
  parser.add_argument("num_workers", default=4,
    help="How many workers will be processing these images at a time? Each worker does batch_size/num_workers images")
  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = parse_args()

  with Pool(processes=args.num_workers) as pool:
    generate_road_sign_data(args)
