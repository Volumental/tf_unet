#!/usr/bin/env python3
import argparse
import os

import tensorflow as tf

from tf_unet import image_util
from tf_unet import unet
from tf_unet import util


def dataset_generator(data_path: str):
    return image_util.ImageDataProvider(os.path.join(data_path, "*.png"),
                                             data_suffix="_color.png",
                                             mask_suffix="_gt.png",
                                             prior_suffix="_prior.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/refinement_segmentation_gt", help="Where the data is.")
    parser.add_argument("--display_step",   default=1,     type=int, help="Number of steps till outputting stats")
    parser.add_argument("--dropout",        default=0.75,  type=int, help="Dropout, probability to keep units")
    parser.add_argument("--batch_size",     default=1,     type=int, help="Number of training examples in each batch")
    parser.add_argument("--training_iters", default=20,    type=int, help="Number of training mini batch iterations per epoch")
    parser.add_argument("--epochs",         default=10,    type=int, help="Number of epochs")
    parser.add_argument("--restore",        default=False, help="If previous model should be restored", action='store_true')
    parser.add_argument("--write_graph",    default=True,  help="If the computation graph should be written as protobuf file to the output path")
    args = parser.parse_args()

    training_generator = dataset_generator(os.path.join(args.data, "train"))
    print("{} examples in training".format(len(training_generator.data_files)))
    print("Image channels: {}".format(training_generator.channels))

    net = unet.Unet(channels=training_generator.channels,
                    n_class=training_generator.n_class,
                    layers=4,
                    features_root=64,
                    cost="dice_coefficient")

    trainer = unet.Trainer(net, batch_size=args.batch_size, optimizer="adam")

    print("Training...")
    path = trainer.train(training_generator, "./unet_trained",
                         training_iters=args.training_iters,
                         epochs=args.epochs,
                         dropout=args.dropout,
                         display_step=args.display_step,
                         restore=args.restore,
                         write_graph=args.write_graph)

    print("Generating test data...")
    testing_generator = dataset_generator(os.path.join(args.data, "test"))
    num_testing_data = len(testing_generator.data_files)
    print("{} examples in test".format(num_testing_data))

    print("Predicting...")
    x_test, y_test = testing_generator(10) # Look at few images
    prediction = net.predict(path, x_test)

    print("Calculating error_rate...")
    error_rate = unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))
    print("Testing error rate: {:.2f}%".format(error_rate))

    print("Visualizing test output...")
    img = util.combine_img_prediction(x_test, y_test, prediction)
    util.save_image(img, "%s/%s.jpg"%(trainer.prediction_path, "_test_output"))


if __name__ == '__main__':
    main()
