# -*- coding:utf-8 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import flags
from generator_model import *

import keras2onnx
import sys

flags.DEFINE_integer("img_size", 256, "Height and width")

flags.DEFINE_string("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint (weight files) path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main():
    A2B_generator = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_generator = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    
    ONNX_A2B_path = "C:/Users/Yuhwan/Documents/New/A2B_generator.onnx"
    ONNX_B2A_path = "C:/Users/Yuhwan/Documents/New/B2A_generator.onnx"

    image = tf.io.read_file("C:/Users/Yuhwan/Pictures/김유환.jpg")
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize(image, [256, 256]) / 127.5 - 1.

    image = tf.expand_dims(image, 0)

    if FLAGS.pre_checkpoint:        # This is just example (how to get the previous weight files in onnx)
        ckpt = tf.train.Checkpoint(A2B_generator)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    onnx_model = keras2onnx.convert_keras(A2B_generator, A2B_generator.name)
    
    content = onnx_model.SerializeToString()

    keras2onnx.save_model(onnx_model, ONNX_A2B_path)

if __name__ == "__main__":
    main()
