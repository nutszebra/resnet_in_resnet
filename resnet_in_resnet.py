import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))


class ResnetInit(nutszebra_chainer.Model):

    def __init__(self, residual_in_channel, transient_in_channel, out_channel=(96, 96), residual_filter_size=(3, 3), transient_filter_size=(3, 3), residual_stride=(1, 1), transient_stride=(1, 1), residual_pad=(1, 1), transient_pad=(1, 1)):
        # out_channel[0]: the number of output channel of residual stream
        # out_channel[1]: the number of output channel of transient stream
        super(ResnetInit, self).__init__()
        modules = []
        modules += [('residual_conv1', L.Convolution2D(residual_in_channel, out_channel[0], residual_filter_size[0], residual_stride[0], residual_pad[0]))]
        modules += [('residual_conv2', L.Convolution2D(residual_in_channel, out_channel[1], residual_filter_size[1], residual_stride[1], residual_pad[1]))]
        modules += [('transient_conv1', L.Convolution2D(transient_in_channel, out_channel[0], transient_filter_size[0], transient_stride[0], transient_pad[0]))]
        modules += [('transient_conv2', L.Convolution2D(transient_in_channel, out_channel[1], transient_filter_size[1], transient_stride[1], transient_pad[1]))]
        modules += [('residual_bn', L.BatchNormalization(out_channel[0]))]
        modules += [('transient_bn', L.BatchNormalization(out_channel[1]))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.residual_in_channel = residual_in_channel
        self.transient_in_channel = transient_in_channel
        self.out_channel = out_channel
        self.residual_filter_size = residual_filter_size
        self.transient_in_channel = transient_filter_size
        self.residual_stride = residual_stride
        self.transient_stride = transient_stride
        self.residual_pad = residual_pad
        self.transient_pad = transient_pad

    def _weight_initialization(self, link):
        link.W.data = self.weight_relu_initialization(link)
        link.b.data = self.bias_initialization(link, constant=0)

    def weight_initialization(self):
        for name, link in self.modules:
            if 'conv' in name:
                self._weight_initialization(link)

    def _count_parameters(self, link):
        return functools.reduce(lambda a, b: a * b, link.W.data.shape)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            if 'conv' in name:
                count += self._count_parameters(link)
        return count

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 == int(np.max([self.residual_stride, self.transient_stride])):
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        x_residual, x_transient = x
        h_residual = self.residual_conv1(x_residual) + self.transient_conv1(x_transient)
        h_residual = h_residual + self.concatenate_zero_pad(self.maybe_pooling(x_residual), h_residual.data.shape, h_residual.volatile, type(h_residual.data))
        h_residual = F.relu(self.residual_bn(h_residual, test=not train))
        h_transient = self.residual_conv2(x_residual) + self.transient_conv2(x_transient)
        h_transient = F.relu(self.transient_bn(h_transient, test=not train))
        return (h_residual, h_transient)


class RiR(nutszebra_chainer.Model):

    def __init__(self, residual_in_channel, transient_in_channel, out_channel=((96, 96), (96, 96)), residual_filter_size=((3, 3), (3, 3)), transient_filter_size=((3, 3), (3, 3)), residual_stride=((1, 1), (1, 1)), transient_stride=((1, 1), (1, 1)), residual_pad=((1, 1), (1, 1)), transient_pad=((1, 1), (1, 1))):
        super(RiR, self).__init__()
        self.residual_in_channel = residual_in_channel
        self.transient_in_channel = transient_in_channel
        self.out_channel = out_channel
        self.residual_filter_size = residual_filter_size
        self.transient_in_channel = transient_filter_size
        self.residual_stride = residual_stride
        self.transient_stride = transient_stride
        self.residual_pad = residual_pad
        self.transient_pad = transient_pad
        modules = []
        for i in six.moves.range(len(out_channel)):
            modules += [('resinit{}'.format(i), ResnetInit(residual_in_channel, transient_in_channel, out_channel[i], residual_filter_size[i], transient_filter_size[i], residual_stride[i], transient_stride[i], residual_pad[i], transient_pad[i]))]
            residual_in_channel = out_channel[i][0]
            transient_in_channel = out_channel[i][1]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 == int(np.max([self.residual_stride, self.transient_stride])):
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        h = x
        for i in six.moves.range(len(self.out_channel)):
            h = self['resinit{}'.format(i)](h, train)
        res_h, trans_h = h
        res_h = res_h + self.concatenate_zero_pad(self.maybe_pooling(x[0]), res_h.data.shape, res_h.volatile, type(res_h.data))
        return (res_h, trans_h)


class ResnetInResnet(nutszebra_chainer.Model):

    def __init__(self, category_num, initial_channel=96):
        self.category_num = category_num
        super(ResnetInResnet, self).__init__()
        # conv
        base = int(initial_channel / 2.)
        modules = []
        modules += [('conv1_1', Conv_BN_ReLU(3, base, 3, 1, 1))]
        modules += [('conv1_2', Conv_BN_ReLU(3, base, 3, 1, 1))]
        modules += [('rir1', RiR(base, base, ((base, base), (base, base))))]
        modules += [('rir2', RiR(base, base, ((base, base), (base, base))))]
        modules += [('rir3', RiR(base, base, ((2 * base, 2 * base), (2 * base, 2 * base)), residual_stride=((1, 1), (2, 2)), transient_stride=((1, 1), (2, 2))))]
        modules += [('rir4', RiR(2 * base, 2 * base, ((2 * base, 2 * base), (2 * base, 2 * base))))]
        modules += [('rir5', RiR(2 * base, 2 * base, ((2 * base, 2 * base), (2 * base, 2 * base))))]
        modules += [('rir6', RiR(2 * base, 2 * base, ((4 * base, 4 * base), (4 * base, 4 * base)), residual_stride=((1, 1), (2, 2)), transient_stride=((1, 1), (2, 2))))]
        modules += [('rir7', RiR(4 * base, 4 * base, ((4 * base, 4 * base), (4 * base, 4 * base))))]
        modules += [('rir8', RiR(4 * base, 4 * base, ((4 * base, 4 * base), (4 * base, 4 * base))))]
        modules.append(('conv2', Conv_BN_ReLU(int(8 * base), category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'resnet_in_resnet_{}_{}'.format(category_num, initial_channel)

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        h = self.conv1_1(x, train), self.conv1_2(x, train)
        for i in six.moves.range(1, 8 + 1):
            h = self['rir{}'.format(i)](h, train)
        h = F.concat(h)
        h = self.conv2(h, train)
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
