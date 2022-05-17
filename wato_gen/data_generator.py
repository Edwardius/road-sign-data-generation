import argparse
import glob
import os
from pathlib import Path
from functools import partial
from PIL import Image, ImageEnhance
import random
import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))

from multiprocessing import Pool, Lock

# classes for YOLOv5, the first few are from the coco dataset
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush', 'do_not_enter', 'left_turn_only_arrow', 'pedestrian_xing', 'right_turn_only_arrow', 'speed_limit_10', 
  'speed_limit_20', 'speed_limit_5', 'handicap_parking', 'left_turn_only_words', 'P_parking', 'right_turn_only_words', 
  'speed_limit_15', 'speed_limit_25']

def write_annotations(f, class_name, x, y, w_f, h_f, w_b, h_b):
  ''' Write annotation to a new label file (.txt). 
      Format: class_index x_centre y_centre width height
      all values normalized from 0 to 1
  '''
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
  lock.acquire()
  try:
    f.write("{} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(class_index, x_centre, y_centre, width, height))
  finally:
    lock.release() 

  return

def combine_images(background, foreground, min_s, min_a):
  # Combine the foreground and background images
  w_b, h_b = background.size
  w_f, h_f = foreground.size

  # Resize foreground image, minimum dimensions must be min_size% the original
  ratio_f = h_f/w_f

  if int(w_f*min_s) < w_f:
    rand_w_f = random.randint(int(w_f*min_s), w_f)
  else:
    rand_w_f = random.randint(w_f, int(w_f*min_s))

  rand_h_f = int(rand_w_f*ratio_f)

  f_new_size = (rand_w_f, rand_h_f)
  foreground = foreground.resize(f_new_size)
  w_f, h_f = foreground.size
  
  # Paste onto background
  # x and y will only ever paste a minimum of min_appearance% * w_f, min_appearance% * h_f 
  # the original sign except when the minimum size is bigger than 1
  x = random.randint(0, w_b-w_f) + random.randint(int(min_a*w_f), int(((1+(1-min_a))*w_f))) - w_f
  y = random.randint(0, h_b-h_f) + random.randint(int(min_a*h_f), int(((1+(1-min_a))*h_f))) - h_f

  background.paste(foreground, (x, y), foreground)

  return x, y, w_f, h_f, w_b, h_b

def process_data(data, output_image, output_label, min_size, min_appearance, glare_factor):
  ''' Load paths, augment images and labels, combine and save them to the output directory
  '''
  # Load paths
  background = Image.open(data["image_path"])
  label_name = os.path.basename(data["label_path"])
  labels = open(data["label_path"])

  # create new label for the data
  f = open(os.path.join(output_label, label_name), "w+")

  # copy over the labels from the original data
  for line in labels:
    f.write(line)
  
  for i in range(len(data["road_sign_names"])):
    foreground = Image.open(data["road_sign_paths"][i])
    class_name = data["road_sign_names"][i]

    # combine the images and update the label
    x, y, w_f, h_f, w_b, h_b = combine_images(background, foreground, min_size, min_appearance)
    write_annotations(f, class_name, x, y, w_f, h_f, w_b, h_b)
  
  # add glare
  if os.path.exists(data["glare_path"]):
    glare = Image.open(data["glare_path"])
    enhancer = ImageEnhance.Brightness(glare)

    glare = enhancer.enhance(glare_factor)
    glare = glare.convert("L")
    
    # Combine the glare and background images
    w_b, h_b = background.size
    glare = glare.resize((w_b, h_b))
    white = Image.new("RGB", (w_b, h_b), (255, 255, 255))
    background.paste(white, (0, 0), glare) 

  background.save(os.path.join(output_image, os.path.basename(data["image_path"])))

  return 

def allocate_work(args, data):
  ''' Allocates batch to the workers. Data consists of 
      [{[road_sign_paths], [road_sign_names], image_path, label_path, glare_path}, ...]
  '''
  # set a couple of parameters alongside the single data iterable we pass in
  output_path = os.path.join(dirname, args.export)

  if not os.path.exists(os.path.join(output_path, "images")):
    os.mkdir(os.path.join(output_path, "images"))

  if not os.path.exists(os.path.join(output_path, "labels")):
    os.mkdir(os.path.join(output_path, "labels"))
   
  partial_func = partial(process_data, 
    output_image = os.path.join(output_path, "images"), 
    output_label = os.path.join(output_path, "labels"),  
    min_size=args.min_size, min_appearance=args.min_appearance, glare_factor=args.glare_factor)

  # we split the work up to multiple workers, hopefully this distributes the load...
  try:
    pool.map(partial_func, data)
  except KeyboardInterrupt:
    print("....\nCaught KeyboardInterrupt, terminating workers")
    pool.terminate()
    pool.join()

  return

def prep_data(args, data, coco_img, coco_lab, road_sign_classes, road_sign_dir, glare_dir):
  # glare
  glare_path = ""
  if random.uniform(0, 1) < args.glare_chance and os.path.exists(glare_dir):
    # we choose the directory through randomly generating a glare filename
    glare_path = os.path.join(glare_dir, 'test/occlusion_images', '{0:05d}.png'.format(random.randint(0, 12450)))

  # folded, gaussian rounded to the nearest integer
  num_road_signs = round(random.gauss(args.mu, args.sigma))
  if num_road_signs < 0:
    num_road_signs = -num_road_signs

  # pick a random class of road sign and a random augmented image of that road sign
  road_signs = []
  road_sign_names = []
  for i in range(num_road_signs):
    road_sign_name = road_sign_classes[random.randint(0, len(road_sign_classes) - 1)]
    road_sign_names.append(road_sign_name)

    # we choose the directory through randomly generating a filename :)
    p = ""
    while not os.path.exists(p):
      p = os.path.join(road_sign_dir, road_sign_name, 'aug{0:05d}.png'.format(random.randint(0, args.max_num_augs)))

    road_signs.append(p)

  data.append({"road_sign_paths": road_signs, "road_sign_names": road_sign_names, 
          "image_path": coco_img, "label_path": coco_lab, "glare_path": glare_path})

  return

def generate_road_sign_data(args):
  ''' Generate synthetic dataset according to given args
      we assume that the number of augmented images is not bigger
      than the coco training dataset (~581000 images)
      Labeling follows the requirements for Yolov5
  '''
  # completing relative paths
  if not os.path.isabs(args.coco):
    print("Road Sign path is not absolute, attempting to complete as relative path")
    coco_dir = os.path.join(dirname, args.coco)

  if not os.path.isabs(args.road_signs):
    print("Road Sign path is not absolute, attempting to complete as relative path")
    road_sign_dir = os.path.join(dirname, args.road_signs)

  if not os.path.isabs(args.glare):
    print("Glare path is not absolute, attempting to complete as relative path")
    glare_dir = os.path.join(dirname, args.glare)

  # names of all the possible road signs classes we can choose from
  road_sign_classes = os.listdir(road_sign_dir)
  
  # iterator for coco images and their labels
  # iglob doesn't have a specific order when iterating, so we have to adapt the labels path to it
  coco_image_iterator = glob.iglob(os.path.join(coco_dir, 'images', 'train2017', '*.jpg'))

  # what we will be sending into the pool for multiprocessing, will be a list of dictionaries
  data_iterable = []

  # iterate through all the coco images, giving each a random number of road signs and optionally glare
  finish_iteration = False
  counter = 0
  pbar = tqdm(total=len(os.listdir(os.path.join(coco_dir, 'images', 'train2017'))))
  while not finish_iteration:
    try:
      coco_image_path = next(coco_image_iterator)
    except StopIteration or KeyboardInterrupt:
      finish_iteration = True
      data_iterable.clear()
    else:
      coco_label_path = os.path.join(coco_dir, 'labels', 'train2017', 
        '{}.txt'.format(Path(os.path.basename(coco_image_path)).stem))

      if not os.path.exists(coco_label_path):
        continue
      
      counter += 1
      pbar.update(1)
      if counter > 5: finish_iteration = True
      # prep data_iterable with the data needed to generate a new image
      prep_data(args, data_iterable, coco_image_path, coco_label_path, road_sign_classes, road_sign_dir, glare_dir)

      # send data iterable in for processing once batch_size is reached, 
      # saves new image and label in args.export
      if len(data_iterable) is args.batch_size:
          allocate_work(args, data_iterable)

          # empty data_iterable for next round of processing
          data_iterable.clear()

  data_iterable.clear()
  pbar.close()

  return

def parse_args():
  '''Parse input arguments
  '''
  parser = argparse.ArgumentParser(description="Creates a dataset of road signs")

  # Path Parameters
  parser.add_argument("--coco", required=True,
    help="The coco directory which contains the images and labels.")
  parser.add_argument("--export", required=True,
    help="The directory where images and labels will be created.")
  parser.add_argument("--road_signs", required=True,
    help="Directory of WATonomous' road signs data.")
  parser.add_argument("--glare", default="", type=str,
    help="Directory of glares")

  # Distribution Parameters
  parser.add_argument("--mu", default=1.5, type=float, 
    help="Mean number of signs in each image")
  parser.add_argument("--sigma", default=0.8, type=float, 
    help="Standard devation of signs in each image. Note this is a Folded Gaussian")

  # Generation Parameters
  parser.add_argument("--batch_size", default=2, type=int,
    help="How many images do we want to process at a time?")
  parser.add_argument("--num_workers", default=1, type=int, 
    help="How many workers will be processing these images at a time? Each worker roughly does batch_size/num_workers images")
  parser.add_argument("--min_size", default=0.1, type=float, 
    help="What is the minimum scale of a road sign if we want to randomly resize it?")
  parser.add_argument("--min_appearance", default=0.9, type=float, 
    help="What is the minimum width/height of a road sign that must appear in the image?")
  parser.add_argument("--glare_chance", default=1.0, type=float, 
    help="How much glare in the dataset? # images with flare = flare_chance * images generated")
  parser.add_argument("--glare_factor", default=0.9, type=float, 
    help="Strength of glare. 1 is opaque whites, 0 is fully transparent.")

  # Cheating Parameters
  ''' max_num_args: in order to not have the algorithm traverse the entire directory of 
      road signs multiple times, a random number generator is used to pick the .png road sign to use
  '''
  parser.add_argument("--max_num_augs", default=2, type=int,
    help="What is the max number of images in a directory of road signs?")
  args = parser.parse_args()  

  return args

def init(l):
  ''' Initialize a global lock at pool creation time. This lock will be used for writing to the labels
  '''
  global lock
  lock = l

if __name__ == '__main__':
  args = parse_args()
  l = Lock()

  ''' Multiple processes could be at risk of producing the same random numbers, 
      but in testing this was not an issue. random.seed(1) maynot be needed here.
  '''
  # random.seed(1) 
  with Pool(processes=args.num_workers, initializer=init, initargs=(l,)) as pool:
    generate_road_sign_data(args)

  pool.close()
  pool.join()