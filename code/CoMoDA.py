# Author: Yevhen Kuznietsov
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from __future__ import absolute_import, division, print_function

import time
import torch.optim as optim
from torch.utils.data import DataLoader
from options import CoMoDAOptions
import json
from layers import *
import os
import datasets
import networks
import random


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class CoMoDA:
    """ Main class for continuous monocular self-supervised depth adaptation
    """
    def __init__(self, options):
        self.opt = options
        self.seed_everything()

        # create dirs for logs and predictions if do not exist
        self.log_path = self.opt.log_dir
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        preds_dir = os.path.join(self.log_path, "preds")
        if not os.path.exists(preds_dir):
            os.mkdir(preds_dir)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        # we don't expect anyone running this on cpu..
        self.device = torch.device("cuda")

        # model initialization
        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, True)
        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers, True,
                                                             num_input_images=self.num_input_frames)
        self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1,
                                                   num_frames_to_predict_for=2)

        for _, m in self.models.items():
            m.to(self.device)
            self.parameters_to_train += list(m.parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = BackprojectDepth(self.opt.batch_size * self.num_scales,
                                                  self.opt.height, self.opt.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.opt.batch_size * (self.num_input_frames - 1) * self.num_scales,
                                    self.opt.height, self.opt.width)
        self.project_3d.to(self.device)

        # save adaptation parameters to the log dir
        self.save_opts()


    def seed_everything(self, seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # for optimized runtime
        if self.opt.benchmarking:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


    def set_adapt(self):
        """Convert all models to adaptation mode: batch norm is in eval mode + frozen params
        """
        # try using frozen batch norm?
        for m in self.models.values():
            m.eval()
            for name, param in m.named_parameters():
                if name.find("bn") != -1:
                   param.requires_grad = False


    def adapt(self):
        """Run adaptation on a list of videos
        """
        img_ext = '.png' if self.opt.png else '.jpg'

        # read the sequence / video names from the corresponding file
        seq_file = self.opt.seq_file
        with open(seq_file, "r") as fseq:
            self.seq_list = [os.path.join(self.opt.seq_dir, l[:-1] + ".txt") for l in fseq.readlines()]

        # read the file names used for experience replay
        experience_filenames = readlines(self.opt.buf_path)

        # initialize dataset / dataloader for experience replay
        experience = datasets.KITTIDataset(self.opt.data_path, experience_filenames, self.opt.height, self.opt.width,
                                           self.opt.frame_ids, self.num_scales, img_ext=img_ext, load_mask=False)
        self.exp_loader = DataLoader(experience, self.opt.batch_size - 1, shuffle=False,
                                     num_workers=self.opt.batch_size - 1, pin_memory=True, drop_last=True)
        self.exp_iterator = iter(self.exp_loader)

        # perform adaptation on every video / sequence separately
        for seq in self.seq_list:
            print("Starting video / sequence: " + seq)
            if self.opt.load_weights_folder is not None:
                self.load_model()

            # initialize dataset / dataloader for the current sequence
            with open(seq, "r") as test_fnames:
                test_files = test_fnames.readlines()
            test_dataset = datasets.KITTIDataset(self.opt.data_path, test_files, self.opt.height, self.opt.width,
                                             self.opt.frame_ids, 4, img_ext=img_ext, load_mask=True)
            self.test_loader = DataLoader(test_dataset, 1, False, num_workers=1, pin_memory=True, drop_last=True)

            self.seq_length = len(test_files)
            self.depth_pred = []

            # run actual adaptation
            self.adapt_seq()

            # save depth predictions
            self.depth_to_save = np.stack(self.depth_pred)
            path_to_save = os.path.join(self.log_path, "preds", seq.split("/")[-1].replace("txt", "npz"))
            np.savez_compressed(path_to_save, data=self.depth_to_save)


    def cat_dict(self, test_input, replay_batch):
        """ Concatenate the elements of test input dictionary
        with the corresponding elements of the past samples dictionary
        """
        res = {}
        for key in test_input:
            if key[0] != "seg":
                res[key] = torch.cat([test_input[key], replay_batch[key]])
            else:
                # the mask is only needed for the frames of the test video
                res[key] = test_input[key]
        return res


    def adapt_seq(self):
        """Run adaptation on a single sequence / video
        """
        print("Adapting")

        # set batch norm of all models to eval mode and freeze its params
        self.set_adapt()

        # init a list to keep timings
        if self.opt.benchmarking:
            runtimes = []

        # iterate over all frames of a test video which have neighbors. The range is [1:-1]
        # (starting with the second frame, finishing with the one before the last)
        for batch_idx, inputs in enumerate(self.test_loader):
            # additional depth predictions are fetched for the first and the last frames of the video
            if batch_idx == 0:
                extra_frame = -1
                print('extra zero frame')
            elif batch_idx == self.seq_length - 1:
                extra_frame = 1
                print('extra last frame')
            else:
                extra_frame = 0

            # fetch the samples from the buffer with past experience and concat them to the test sample
            # concatenation is only for the adaptation with the camera parameters being unchanged
            # otherwise - those samples are processed separately, with losses combined (not implemented here)
            try:
                replay_batch = self.exp_iterator.next()
            except StopIteration:
                self.exp_iterator = iter(self.exp_loader)
                replay_batch = self.exp_iterator.next()
            inputs = self.cat_dict(inputs, replay_batch)

            # transfer the inputs to gpu
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            # start time measurement here
            # todo: this kind of time measurement is not precise, use pytorch internal tools
            if self.opt.benchmarking:
                start = time.time()

            # forward pass
            outputs, losses = self.process_batch(inputs, extra_frame)

            # check if the vehicle moves fast enough, so that the camera translation is not too small
            # if not the case, do not update the model params
            if torch.min(inputs[("trans:0-1")][0], inputs[("trans:1-2")][0]) > self.opt.min_translation:
                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()

            # end time measurement here
            if self.opt.benchmarking:
                end = time.time()
                duration = end - start
                if batch_idx != 0 and batch_idx != self.seq_length - 1:
                    runtimes.append(duration)
                if batch_idx == self.seq_length - 1:
                    avg_runtime = np.mean(np.asarray(runtimes))
                    print("Avg runtime is {:0.4f}s".format(avg_runtime))

            # append the inverse depth predicted for the current frame
            # to the list of predictions for the preceding frames
            if batch_idx == 0:
                self.depth_pred.append(self.extra_output[("disp", 0)][0][0].cpu().detach().numpy())
            self.depth_pred.append(outputs[("disp", 0)][0][0].cpu().detach().numpy())
            if batch_idx == self.seq_length - 1:
                self.depth_pred.append(self.extra_output[("disp", 0)][0][0].cpu().detach().numpy())


    def process_batch(self, inputs, extra_frame):
        """Pass a minibatch through the network and generate images and losses
        """
        # depth inference
        features = self.models["encoder"](inputs["color", 0, 0])
        outputs = self.models["depth"](features)

        # additional predictions for the first and the last frames of video
        if extra_frame != 0:
            extra_features = self.models["encoder"](inputs["color", extra_frame, 0])
            self.extra_output = self.models["depth"](extra_features)

        # pose inference
        outputs.update(self.predict_poses(inputs))

        # image reconstruction
        self.reconstruct_images(inputs, outputs)

        # compute loss
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses


    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        Simplified variant of monodepth2 implementation.
        """
        outputs = {}

        # Here we input all frames to the pose net (and predict all poses) together
        pose_inputs = torch.cat([inputs[("color", i, 0)] for i in self.opt.frame_ids], 1)
        pose_inputs = [self.models["pose_encoder"](pose_inputs)]

        axisangle, translation = self.models["pose"](pose_inputs)

        # transform predicted ego-motion into matrix form
        for i, f_i in enumerate(self.opt.frame_ids[1:]):
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])

        return outputs


    def reconstruct_images(self, inputs, outputs):
        """ Perform novel view synthesis. Runtime-optimized implementation of a similar method from monodepth2 """
        # resize the depth predictions of every scale to the output resolution
        ms_depth = []
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            depth = 1 / disp
            ms_depth.append(depth)

        # scale 0 sample 0, scale 0 sample 1, ... scale 0 sample n, scale 1 sample 0...
        # batch_size x num_scales
        ms_depth = torch.cat(ms_depth, dim=0)
        source_scale = 0

        # perform projection and back-projection without "for loops"
        imgs_to_warp = []
        for frame_id in self.opt.frame_ids[1:]:
            for _ in self.opt.scales:
                imgs_to_warp.append(inputs[("color", frame_id, source_scale)])
        imgs_to_warp = torch.cat(imgs_to_warp, dim=0)

        num_neighbors = self.num_input_frames - 1
        # frame 0 scale 0 sample 0, frame scale 0 sample 1, ... frame 0 scale 0 sample n, frame 0 scale 1 sample 0...
        # num_neighbors x batch_size x num_scales
        cam_points = self.backproject_depth(ms_depth, inputs[("inv_K", source_scale)].
                                            repeat([self.num_scales, 1, 1])).repeat(num_neighbors, 1, 1)
        transforms = torch.cat([outputs[("cam_T_cam", 0, frame_id)].repeat(self.num_scales, 1, 1)
                               for frame_id in self.opt.frame_ids[1:]], dim=0)
        pix_coords = self.project_3d(cam_points, inputs[("K", source_scale)].
                                     repeat([self.num_scales * num_neighbors, 1, 1]), transforms)

        # frame 0 scale 0 sample 0, frame scale 0 sample 1, ... frame 0 scale 0 sample n, frame 0 scale 1 sample 0...
        # num_neighbors x num_scales x batch_size
        warped_imgs = F.grid_sample(imgs_to_warp, pix_coords, padding_mode="border")
        outputs["all_reconstructions"] = warped_imgs


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_losses(self, inputs, outputs):
        """ Compute the total loss
        """
        losses = {}
        total_loss = 0
        num_neighbors = self.num_input_frames - 1

        # compute image reconstruction loss for all scales
        # tensor-based (without for loop) computations allow faster runtime compared to monodepth2
        pred = outputs["all_reconstructions"]
        identity_list = [inputs[("color", frame_id, 0)] for frame_id in self.opt.frame_ids[1:]]
        identity = torch.cat(identity_list)
        reprojection_losses = self.compute_reprojection_loss(pred, inputs[("color", 0, 0)].
                                                             repeat([self.num_scales * num_neighbors, 1, 1, 1]))

        # auto-masking, min loss
        identity_losses = self.compute_reprojection_loss(identity, inputs[("color", 0, 0)].
                                                         repeat([num_neighbors, 1, 1, 1]))
        identity_losses += torch.randn(identity_losses.shape).cuda() * 0.00001

        # frame, scale x sample, h, w
        identity_losses = identity_losses.reshape([num_neighbors, self.opt.batch_size, self.opt.height,
                                                   self.opt.width]).repeat([1, self.num_scales, 1, 1])

        # frame 0 scale 0 sample 0, frame scale 0 sample 1, ... frame 0 scale 0 sample n, frame 0 scale 1 sample 0...
        # frame x scale x sample, ... -> frame, scale x sample, h, w
        reprojection_losses = reprojection_losses.reshape([num_neighbors, self.opt.batch_size * self.num_scales,
                                                           self.opt.height, self.opt.width])

        combined = torch.cat([identity_losses, reprojection_losses], dim=0)

        # scale x sample, h, w
        to_optimise, idxs = torch.min(combined, dim=0)

        # mask out possibly moving objects
        mask_mov = inputs[("seg", 0, 0)].squeeze(1).float()
        ones_mask = torch.ones_like(mask_mov).repeat([self.opt.batch_size - 1, 1, 1])
        mask_mov = torch.cat([mask_mov, ones_mask]).repeat(self.num_scales, 1, 1)
        to_optimise = to_optimise * mask_mov

        total_loss += to_optimise.reshape([self.num_scales, self.opt.batch_size, self.opt.height, self.opt.width]).\
                                          mean([1, 2, 3]).sum()

        # compute smoothness for every scale
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            total_loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

        total_loss /= self.num_scales

        # translation / velocity supervision loss
        translation_loss = torch.abs(torch.norm(outputs[("translation", 0, -1)][:, 0], 2, dim=-1).squeeze() -
                            inputs[("trans:0-1")]) + \
                            torch.abs(torch.norm(outputs[("translation", 0, 1)][:, 0], 2, dim=-1).squeeze() -
                            inputs[("trans:1-2")])

        translation_loss = self.opt.translation_weight * translation_loss
        losses["loss"] = total_loss + translation_loss.mean()
        return losses


    def save_opts(self):
        """Save options to disk
        """
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(self.log_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    def load_model(self):
        """Load a pre-trained model from disk, no optimizer loaded
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder)

        if "weights_" not in self.opt.load_weights_folder:
            weight_dirs = os.listdir(self.opt.load_weights_folder)
            models = []
            for m in weight_dirs:
                if "weights_" in m:
                    models.append(int(m[8:]))

            assert len(models) > 0

            models.sort()
            last_epoch = models[-1]
            self.opt.load_weights_folder = os.path.join(self.opt.load_weights_folder, "weights_" + str(last_epoch))

        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)


options = CoMoDAOptions()
opts = options.parse()


if __name__ == "__main__":
    adaptor = CoMoDA(opts)
    adaptor.adapt()

