"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

#--model models/Hazy2GT_outdoor.pb --input data/outdoor/41.png --output results/outdoor/41_output.png --image_size 512

import tensorflow.compat.v1 as tf
import os
import glob
import numpy as np
import cv2
from model import CycleGAN
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'models/Hazy2GT_outdoor.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'data/outdoor/frame_49758.png', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', 'results/outdoor/frame_49758.png', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')

def benchmark_ots():
  HAZY_PATH = "D:/Datasets/OTS_BETA/haze/"
  SAVE_PATH = "results_reside/"
  hazy_list = glob.glob(HAZY_PATH + "*0.95_0.2.jpg") #specify atmosphere intensity

  #perform inference
  for i in range(1934, len(hazy_list)):
    inference(hazy_list[i], SAVE_PATH)

def benchmark_reside():
  HAZY_PATH = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
  SAVE_PATH = "results_reside/"
  hazy_list = glob.glob(HAZY_PATH + "*.jpeg")

  #perform inference
  for i in range(len(hazy_list)):
    inference(hazy_list[i], SAVE_PATH)

def benchmark_ohaze():
  HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
  SAVE_PATH = "results_ohaze/"
  hazy_list = glob.glob(HAZY_PATH + "*.jpg")

  #perform inference
  for i in range(len(hazy_list)):
    inference(hazy_list[i], SAVE_PATH)


def benchmark_ihaze():
  HAZY_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/hazy/"
  SAVE_PATH = "results_ihaze/"
  hazy_list = glob.glob(HAZY_PATH + "*.jpg")

  # perform inference
  for i in range(len(hazy_list)):
    inference(hazy_list[i], SAVE_PATH)



def benchmark():
  HAZY_PATH= "E:/Hazy Dataset Benchmark/I-HAZE/hazy/"
  GT_PATH = "E:/Hazy Dataset Benchmark/I-HAZE/GT/"
  SAVE_PATH = "results/"
  BENCHMARK_PATH = "results/metrics.txt"
  hazy_list = []
  gt_list = []
  result_list = []

  for (root, dirs, files) in os.walk(HAZY_PATH):
    for f in files:
      file_name = os.path.join(root, f)
      hazy_list.append(file_name)

  for (root, dirs, files) in os.walk(GT_PATH):
    for f in files:
      file_name = os.path.join(root, f)
      gt_list.append(file_name)

  #perform inference
  # for i in range(len(hazy_list)):
  #   inference(hazy_list[i], SAVE_PATH)

  #measure performance
  result_list = glob.glob(SAVE_PATH + "*.jpg")
  # for (root, dirs, files) in os.walk(SAVE_PATH):
  #   for f in files:
  #     file_name = os.path.join(root, f)
  #     result_list.append(file_name)

  #check SSIM
  print(result_list)
  average_SSIM = 0.0
  with open(BENCHMARK_PATH, "w") as f:
    for i in range(len(result_list)):
      input_name = result_list[i].split('\\')[0]
      result_img = cv2.imread(result_list[i])
      gt_img = cv2.imread(gt_list[i])
      gt_img = cv2.resize(gt_img, (256, 256), interpolation = cv2.INTER_CUBIC)

      SSIM = np.round(structural_similarity(result_img, gt_img, multichannel=True),4)
      print("SSIM of " +input_name+ " : ", SSIM, file = f)
      average_SSIM += SSIM

    average_SSIM = average_SSIM / len(result_list) * 1.0
    print("Average SSIM: ", average_SSIM, file = f)

def inference(input_img_path, result_save_path):
  graph = tf.Graph()
  #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

  with graph.as_default():
    #with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
    with tf.gfile.FastGFile(input_img_path, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')
  with tf.Session(graph=graph) as sess:
    generated = output_image.eval()
    input_name = input_img_path.split('\\')[1]
    #with open(FLAGS.output, 'wb') as f:
    with open(result_save_path + input_name, 'wb') as f:
      f.write(generated)

def main(unused_argv):
  #inference(FLAGS.input, FLAGS.output)
  #benchmark()
  #benchmark_reside()
  #benchmark_ohaze()
  #benchmark_ots()
  benchmark_ihaze()

if __name__ == '__main__':
  tf.app.run()
