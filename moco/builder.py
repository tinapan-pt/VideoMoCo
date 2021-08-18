import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator_Mask(nn.Module):
    '''
    generator for the mask:
    topk can be adaptive for the training process
    '''

    def __init__(self, C_in, embedding_dim, hidden_dim, output_dim,
                 n_layers=3, kernel_size=3, padding=1, use_bidirectional=True, use_dropout=False):
        super().__init__()
        self.Conv_BN = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, bias=False, dilation=2),
            nn.BatchNorm2d(C_in, eps=1e-3, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=padding)
        )
        in_channel = C_in
        depth = in_channel
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.conv1 = nn.Sequential(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(depth, output_dim, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ReLU(inplace=True))
        self.pool = nn.AvgPool2d(2)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim // 2,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0., batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Softmax())
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)

        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.Conv_BN(x)
        # ASPP
        size = x.shape[2:]
        b, c = x.shape[0], x.shape[1]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        net = self.conv2(net)
        for i in range(3):
            net = self.conv1(net)
            net = self.conv1(net)
            net = self.pool(net)
        lstm_input = net.view(b, self.output_dim, -1)
        output, (hidden, cell) = self.rnn(lstm_input)

        return self.fc(output[:, -1, :])

class VideoTopk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, im_q, list_out):
        b, c, n, h, w = im_q.shape
        values, indices = list_out.topk(8, dim=1, largest=False, sorted=False)
        im_q_fake = im_q.clone().detach()
        for i in range(b):
            for j in indices[i]:
                im_q_fake[i, :, j, :, :] = torch.mean(im_q_fake[i, :, j, :, :])
        return im_q_fake

    @staticmethod
    def backward(ctx, grad_output):
        grad_im = None
        grad_list = torch.sum(grad_output, dim=(1, 3, 4))
        return grad_im, grad_list

class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator, self).__init__()
        self.mask_generator = Generator_Mask(C_in=32 * 3, embedding_dim=169, hidden_dim=128, output_dim=32)
        self.video_topk = VideoTopk()

    def forward(self, im_q):
        # compute query features
        b, c, n, h, w = im_q.shape
        reshape = im_q.view(b, -1, h, w)
        list_out = self.mask_generator(reshape)
        im_q_fake = self.video_topk.apply(im_q,list_out)
        return im_q_fake

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.count = torch.zeros(K, dtype=torch.int16).cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        self.count += 1
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        self.count[ptr:ptr + batch_size] = 1
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q_fake, im_q, im_k, t):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        torch.autograd.set_detect_anomaly(True)
        q_real = self.encoder_q(im_q)
        q_real = nn.functional.normalize(q_real, dim=1)
        q_fake = self.encoder_q(im_q_fake)  # queries: NxC
        q = nn.functional.normalize(q_fake, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        weight = t ** (1.0 * self.count).cuda()
        weight = torch.mul(self.queue, weight).cuda()
        l_neg = torch.einsum('nc,ck->nk', [q, weight.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return q, q_real, logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
