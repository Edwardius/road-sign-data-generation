import argparse
from cProfile import label
import glob  
import os
from pathlib import Path
from functools import partial
from PIL import Image
import random

dirname = os.path.dirname(os.path.abspath(__file__))

from multiprocessing import Pool

# classes for YOLOv5, the first few are from the coco dataset
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush', 'do_not_enter', 'handicap_parking', 'left_turn_only_arrow', 'left_turn_only_words']

def write_annotations(output_label, label_name, labels, class_name, x, y, w_f, h_f, w_b, h_b):
  ''' Write annotation to a new label file (.txt). 
      Format: class_index x_centre y_centre width height
      all values normalized from 0 to 1
  '''
  f = open(os.path.join(output_label, label_name), "w+")

  # copy over the labels from the original data
  for line in labels:
    f.write(line)

  # get index of the class
  class_index = class_names.index(class_name)

  # for images that are off the screen...
  # too far right or too far down
  if x + w_f > w_b:
    w_f = w_b - x
  if y + h_f > h_b:
    h_f = h_b - y

  # too far left or too far up
  if x < 0:
    w_f = w_f + x
    x = 0
  if y < 0:
    h_f = h_f + y
    y = 0

  # get centres and normalize
  x_centre = (x + w_f/2)/w_b
  y_centre = (y + h_f/2)/h_b
  width = w_f/w_b
  height = h_f/h_b

  # write to txt 
  f.write("{} {:0.6f} {:0.6f} {:0.6f} {:0.6f}".format(class_index, x_centre, y_centre, width, height))

  return

def combine_data(data, output_image, output_label, min_size, min_appearance):
  ''' Load paths, augment images and labels, save them to the output directory
  '''
  # Load paths
  background = Image.open(data["image_path"])
  foreground = Image.open(data["road_sign_path"])
  class_name = data["road_sign_class"]
  labels = open(data["label_path"])
  label_name = os.path.basename(data["label_path"])
 
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

  write_annotations(output_label, label_name, labels, class_name, x, y, w_f, h_f, w_b, h_b)

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

  # we split the work up to multiple workers, hopefully this distributes the load...
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
  # iglob doesn't have a specific order when iterating, so we have to adapt the labels path to it
  coco_image_iterator = glob.iglob(os.path.join(coco_dir, 'images', 'train2017', '*.jpg'))

  # what we will be sending into the pool for multiprocessing
  data_iterable = []

  # for each of the road sign categories, this will make sure that every road sign 
  # augment is used
  road_sign_classes = os.listdir(road_sign_dir)
  
  for road_sign_class in road_sign_classes:
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
        coco_image_path = next(coco_image_iterator)
        coco_label_path = os.path.join(coco_dir, 'labels', 'train2017', '{}.txt'.format(Path(os.path.basename(coco_image_path)).stem))

        data_iterable.append({"road_sign_path": road_sign, "road_sign_class": road_sign_class, 
          "image_path": coco_image_path, "label_path": coco_label_path})

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
  parser.add_argument("max_num_instances", default=3, type=float, 
    help="What is the maximum number of instances of road signs in an image?")
  parser.add_argument("flare_num", default=0.1, type=float, 
    help="How much flare in the dataset? # images with flare = flare_num * images generated")
  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = parse_args()

  with Pool(processes=args.num_workers) as pool:
    generate_road_sign_data(args)