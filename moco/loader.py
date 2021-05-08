import random
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
import torch
import kornia
import torchvision.transforms as transforms


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.], img_size=112):
        self.sigma = sigma
        self.radius = int(0.1*img_size)//2*2+1

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        gauss = kornia.filters.GaussianBlur2d((self.radius, self.radius), (sigma, sigma))
        return gauss(x)


class MoCoAugment(object):

    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment = transforms.Compose(
            [
                kornia.augmentation.RandomGrayscale(p=0.2),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.4),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


class MoCoAugmentV2(object):
    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment_v2 = transforms.Compose(
            [
                transforms.RandomApply([
                    kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                kornia.augmentation.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.], crop_size)], p=0.5),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video,
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


class RandomTwoClipSampler(Sampler):
    """
    Samples two clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select two clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 2:
                sampled = [s, s]
            else:
                sampled = torch.randperm(length)[:2] + s
                sampled = sampled.tolist()
            s += length
            idxs.append(sampled)
        # shuffle all clips randomly
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)


class DummyAudioTransform(object):
    """This is a dummy audio transform.

    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible

    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)
