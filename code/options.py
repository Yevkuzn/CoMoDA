# Author: Yevhen Kuznietsov
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from __future__ import absolute_import, division, print_function
import argparse


class CoMoDAOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="CoMoDA options")

        # Paths
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to data",
                                 default="...")
        self.parser.add_argument("--seq_file",
                                 type=str,
                                 help="path to the file with the sequence names",
                                 default="...")
        self.parser.add_argument("--seq_dir",
                                 type=str,
                                 help="path to the dir with the filenames files for every seq",
                                 default="...")
        self.parser.add_argument("--buf_path",
                                 type=str,
                                 help="path to a pregenerated list of samples to use for experience replay",
                                 default="...")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="dir to write logs / save adapted predictions",
                                 default="...")
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="path to the dir with the pretrained model weights",
                                 default="...")

        # Model options
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--png",
                                 help="if set, use PNG instead of JPEG."
                                      "JPEG must be faster to load, but some details might be not preserved",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--translation_weight",
                                 type=float,
                                 help="velocity supervision term weight",
                                 default=0.005)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="pretrained model parts to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])


        # Adaptation options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size (1 + the number of smaples drawn from the replay buffer)",
                                 default=4)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate for adaptation",
                                 default=0.0001)
        self.parser.add_argument("--min_translation",
                                 type=float,
                                 help="translation threshold to optimize the model parameters",
                                 default=0.2)

        # Other
        self.parser.add_argument("--benchmarking",
                                 help="if set, run in benchmarking mode."
                                      "The runtime is faster, but the results are slightly more random",
                                 action="store_true")



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
