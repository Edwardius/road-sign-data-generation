import argparse
import glob  
import os
from functools import partial
from PIL import Image
import random

dirname = os.path.dirname(os.path.abspath(__file__))

from multiprocessing import Pool

def write_annotations(labels, class_name, x, y, w_f, h_f):
  
  return

       
def combine_data(data, output_image, output_label, min_size, min_appearance):
  ''' Load paths, augment images and labels, save them to the output directory
  '''
  # Load paths
  background = Image.open(data["image_path"])
  foreground = Image.open(data["road_sign_path"])
  class_name = data["road_sign_class"]
  labels = open(data["label_path"])
 
  # Combine the foreground and background images
  w_b, h_b = background.size
  w_f, h_f = foreground.size

  # Resize foreground image, minimum dimensions must be min_size% the original
  ratio_f = h_f/w_f
  rand_w_f = random.randint(int(w_f*min_size), w_f)
  rand_h_f = int(rand_w_f*ratio_f)

  f_new_size = (rand_w_f, rand_h_f)
  foreground = foreground.resize(f_new_size)
  w_f, h_f = foreground.size
  
  # Paste onto background
  # x and y will only ever paste a minimum of min_appearance% * w_f, min_appearance% * h_f 
  # the original sign
  x = random.randint(0, w_b-w_f) + random.randint(int(min_appearance*w_f), int(((1+(1-min_appearance))*w_f))) - w_f
  y = random.randint(0, h_b-h_f) + random.randint(int(min_appearance*h_f), int(((1+(1-min_appearance))*h_f))) - h_f
  background.paste(foreground, (x, y), foreground)
  background.save(os.path.join(output_image, os.path.basename(data["image_path"])))

  write_annotations(labels, class_name, x, y, w_f, h_f)

  return 

def process_synth_data(args, data):
  ''' Allocates batch to the workers. Data consists of 
      [{road_sign_path, road_sign_class, image_path, label_path}, ...]
  '''
  # set a couple of parameters alongside the single data iterable we pass in
  output_path = os.path.join(dirname, args.export)

  if not os.path.exists(os.path.join(output_path, "images")):
    os.mkdir(os.path.join(output_path, "images"))

  if not os.path.exists(os.path.join(output_path, "labels")):
    os.mkdir(os.path.join(output_path, "labels"))
   
  partial_func = partial(combine_data, 
    output_image = os.path.join(output_path, "images"), 
    output_label = os.path.join(output_path, "labels"),  
    min_size=args.min_size, min_appearance=args.min_appearance)

  try:
    pool.map(partial_func, data)
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

  # for each of the road sign categories, this will make sure that every road sign 
  # augment is used
  for x in os.listdir(road_sign_dir):
    road_sign_class = x

    # iterator over the folder of augmented road_signs
    road_sign_iterator = glob.iglob(os.path.join(road_sign_dir, road_sign_class, '*.png'))
    finish_iteration = False

    # iterate over all the sign images in the folder 
    while not finish_iteration:
      try:
        road_sign = next(road_sign_iterator)
      except StopIteration:
        finish_iteration = True
      else:
        data_iterable.append({"road_sign_path": road_sign, "road_sign_class": road_sign_class, 
          "image_path": next(coco_image_iterator), "label_path": next(coco_label_iterator)})

        if len(data_iterable) is args.batch_size:
          # send data_iterable in for processing
          process_synth_data(args, data_iterable)

          # empty data_iterable
          data_iterable.clear()
  
  # if we want multiple road signs in one image
  # iterate through new images and labels at random, add some extra random road signs to some of them, 
  # delete the prev

  # if we want flare 
  # iterate through images and labels at random, add some flare, delete the prev

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
  parser.add_argument("batch_size", default=64, type=int,
    help="How many images do we want to process at a time?")
  parser.add_argument("num_workers", default=4, type=int, 
    help="How many workers will be processing these images at a time? Each worker roughly does batch_size/num_workers images")
  parser.add_argument("min_size", default=0.1, type=float, 
    help="What is the minimum scale of a road sign if we want to randomly resize it?")
  parser.add_argument("min_appearance", default=0.9, type=float, 
    help="What is the minimum width/height of a road sign that must appear in the image?")
  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = parse_args()

  with Pool(processes=args.num_workers) as pool:
    generate_road_sign_data(args)