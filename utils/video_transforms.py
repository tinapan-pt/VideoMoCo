import random
import numpy as np
import cv2


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video_clip):

        if random.random() < self.p:
            # t x h x w
            #print("flip")
            flip_video_clip = np.flip(video_clip, axis=2).copy()

            return flip_video_clip

        return video_clip


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, video_clip):

        h, w = video_clip.shape[1:3]
        new_h, new_w = self.output_size

        h_start = random.randint(0, h-new_h)
        w_start = random.randint(0, w-new_w)

        rnd_crop_video_clip = video_clip[:, h_start:h_start+new_h,
                                 w_start:w_start+new_w, :]

        return rnd_crop_video_clip


class CenterCrop(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, video_clip):

        h, w = video_clip.shape[1:3]

        new_h, new_w = self.output_size

        h_start = int((h - new_h) / 2)
        w_start = int((w- new_w) / 2)

        center_crop_video_clip = video_clip[:, h_start:h_start + new_h,
                                    w_start:w_start + new_w, :]

        return center_crop_video_clip


class ClipResize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, video_clip):
        rsz_video_clip = []
        new_h, new_w = self.output_size

        for frame in video_clip:
            rsz_frame = cv2.resize(frame, (new_w, new_h))
            rsz_video_clip.append(rsz_frame)

        return np.array(rsz_video_clip)



class ToTensor(object):
    """
    change input channel
    D x H x W x C ---> C x D x H x w
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, sample):
        video_clip = sample

        video_clip = np.transpose(video_clip, (3, 0, 1, 2))

        return video_clip

