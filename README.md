# RoadSignDataGerenation
WATonomous is looking into simple synthetic data generation for road sign detection.

`wato_gen` contains WATonomous' proprietary synthetic data generator for YOLOv5. It pastes a random number of augmented road signs onto each of the images in the coco dataset. The frequency of signs on an image is naively distributed as a folded gaussian.

The data generator takes advantage of multiprocessing to efficiently synthesize batches of data.

Labelling scheme is following the requirements of https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

## Setup
---
Download the coco dataset, must be in the format of https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh
```
./wato_gen/get_coco_dataset.sh
```
Download the glare dataset through https://bigmms.github.io/chen_tits21_dataset/

Obtain a dataset of augmented traffic signs, WATonomous will keep this a secret c:

## Usage
---
Simply run `wato_gen/data_generator.py`. There are some parameters you have to enter along with it. Here is a list of them, most have defaults.
```
usage: data_generator.py [-h] --coco COCO --export EXPORT --road_signs ROAD_SIGNS [--glare GLARE] [--mu MU] [--sigma SIGMA]
                         [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--min_size MIN_SIZE] [--min_appearance MIN_APPEARANCE]
                         [--max_num_instances MAX_NUM_INSTANCES] [--glare_chance GLARE_CHANCE] [--glare_factor GLARE_FACTOR]
                         [--max_num_augs MAX_NUM_AUGS]

Creates a dataset of road signs

required arguments:
  -h, --help            show this help message and exit
  --coco COCO           The coco directory which contains the images and labels.
  --export EXPORT       The directory where images and labels will be created.
  --road_signs ROAD_SIGNS  WATonomous' road signs data.

optional arguments:
  --glare GLARE         WATonomous' road signs data.
  --mu MU               Mean number of signs in each image
  --sigma SIGMA         Standard devation of signs in each image. Note this is a Folded Gaussian
  --batch_size BATCH_SIZE
                        How many images do we want to process at a time?
  --num_workers NUM_WORKERS
                        How many workers will be processing these images at a time? Each worker roughly does batch_size/num_workers images
  --min_size MIN_SIZE   What is the minimum scale of a road sign if we want to randomly resize it?
  --min_appearance MIN_APPEARANCE
                        What is the minimum width/height of a road sign that must appear in the image?
  --glare_chance GLARE_CHANCE
                        How much glare in the dataset? # images with flare = flare_chance * images generated
  --glare_factor GLARE_FACTOR
                        Strength of glare. 1 is opaque whites, 0 is fully transparent.
  --max_num_augs MAX_NUM_AUGS
                        What is the max number of images in a directory of road signs?

```
`-h` might say something different, but at this moment, the above is the list of arguments you should reference.