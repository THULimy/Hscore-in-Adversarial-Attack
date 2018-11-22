"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.python.platform import flags
from torch.autograd import Variable
from torchvision import datasets, transforms

from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

FLAGS = flags.FLAGS


class nnModel(nn.Module):
    """ Basic MNIST model from github
    https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
    """

    def __init__(self):
        super(nnModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding

        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class aceModel_f(nn.Module):
    """ Basic MNIST model from github
    https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
    """

    def __init__(self):
        super(aceModel_f, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

    def forward(self, x):
        f = F.max_pool2d(F.relu(self.conv1(x)), 2)
        f = F.max_pool2d(F.relu(self.conv2(f)), 2)
        f = f.view(-1, 64 * 7 * 7)  # reshape Variable
        f = F.relu(self.fc1(f))
        return f

class aceModel_g(nn.Module):
    """ Basic MNIST model from github
    https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
    """
    def __init__(self):
        super(aceModel_g, self).__init__()
        self.fc1 = nn.Linear(10, 128)

    def forward(self, y):
        g = F.relu(self.fc1(y))
        return g

class aceModel(nn.Module):
    """ Basic MNIST model from github
    https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
    """
    def __init__(self):
        super(aceModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding

        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def neg_hscore(f,g):
	f0 = f - torch.mean(f,0)
	g0 = g - torch.mean(g,0)
	corr = torch.mean(torch.sum(f0*g0,1))
	cov_f = torch.mm(torch.t(f0),f0) / (f0.size()[0]-1.)
	cov_g = torch.mm(torch.t(g0),g0) / (g0.size()[0]-1.)
	return - corr + torch.trace(torch.mm(cov_f, cov_g)) / 2.

def mnist_tutorial(nb_epochs=1, batch_size=128, train_end=-1, test_end=-1,
                   learning_rate=0.001):
    """
    MNIST cleverhans tutorial
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # Train a pytorch MNIST model

    # seed = 1
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    train = 0
    model_ace = aceModel()
    model_nn = nnModel()
    model_f = aceModel_f()
    model_g = aceModel_g()
    if torch.cuda.is_available():
        model_nn = model_nn.cuda()
        model_g = model_g.cuda()
        model_f = model_f.cuda()
        model_ace = model_ace.cuda()
    report = AccuracyReport()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size)

    # Truncate the datasets so that our test run more quickly
    train_loader.dataset.train_data = train_loader.dataset.train_data[
                                      :train_end]
    test_loader.dataset.test_data = test_loader.dataset.test_data[:test_end]

    # Train our model
    optimizer_nn = optim.Adam(model_nn.parameters(),lr=learning_rate)
    # optimizer_nn = optim.Adam(model_nn.parameters(), lr=learning_rate)
    optimizer_ace = optim.Adam(list(model_f.parameters())+list(model_g.parameters()), lr=learning_rate)
     
    if train:
        for epoch in range(nb_epochs):
            train_total = 0
            correct = 0
            step = 0
            py = np.zeros((1,10))
            loss_nn_total=0
            for xs, ys in train_loader:
                ys_1hot = torch.zeros(len(ys), 10).scatter_(1, ys.resize(len(ys),1), 1)
                # print(ys,ys.size(),ys.type())
                xs, ys = Variable(xs), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys ,ys_1hot = xs.cuda(), ys.cuda(), ys_1hot.cuda()
                optimizer_nn.zero_grad()
                optimizer_ace.zero_grad()
                logits_nn = model_nn(xs)
                # pred = torch.max(logits_nn,1)[1]
                # acc = (pred==ys).sum()
                print(xs[-1])
                f = model_f(xs)
                g = model_g(ys_1hot)
                loss_ace = neg_hscore(f,g)
                loss_ace.backward()
                loss_nn = F.cross_entropy(logits_nn, ys)
                loss_nn.backward()  # calc gradients
                # print(loss_nn)
                optimizer_nn.step()
                optimizer_ace.step()  # update gradients
                py = py + torch.sum(ys_1hot,0).cpu().numpy()
                train_total += len(xs)
            print('Epoch {} finished.'.format(epoch))
            print("loss_nn:{},loss_ace:{}".format(loss_nn,loss_ace))
            # preds_np = preds.data.cpu().numpy()
            # correct += (np.argmax(preds_np, axis=1) == ys).sum()
            # step += 1
            # if total % 1000 == 0:
            #     acc = float(correct) / total
            #     print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
            #     total = 0
            #     correct = 0
            py = py.astype(float) / train_total
            # Evaluate on clean data
            total = 0
            correct_ace = 0
            correct_nn = 0
            if torch.cuda.is_available():
                eye = torch.eye(10).cuda()
            else:
                eye = torch.eye(10)
            g_test = model_g(eye).data.cpu().numpy()
            g_test = g_test - np.mean(g_test, axis = 0)
            for xs, ys in test_loader:
                xs, ys = Variable(xs), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()

                logits_nn = model_nn(xs).data.cpu().numpy()
                # logits_nn_np = logits_nn.data.cpu().numpy()
                f_test = model_f(xs).data.cpu().numpy()
                f_test = f_test - np.mean(f_test, axis = 0)
                
                # py = np.mean(y_train, axis = 0)
                pygx = py * (1 + np.matmul(f_test, g_test.T))
                # ace_acc = np.mean(np.argmax(pygx, axis = 1) == np.argmax(y_test, axis = 1))

                correct_ace += (np.argmax(pygx, axis = 1) == ys).sum()
                correct_nn += (np.argmax(logits_nn, axis=1) == ys).sum()
                total += len(xs)

            nn_acc = float(correct_nn) / total
            ace_acc = float(correct_ace) / total

            print('NN test accuracy: %.2f%%' % (nn_acc * 100))
            print('ACE test accuracy: %.2f%%' % (ace_acc * 100))

        print("py:{},pygx:{}".format(py,pygx))
        model_f_dict = model_f.state_dict()
        model_ace_dict = model_ace.state_dict()
        model_f_dict = {key: value for key, value in model_f_dict.items() if key in model_ace_dict}
        if torch.cuda.is_available():
            model_f_dict['fc2.weight'] = torch.from_numpy((py * g_test.T).T).cuda()
            model_f_dict['fc2.bias'] = torch.from_numpy(py).view(10).cuda()
        else:
            model_f_dict['fc2.weight'] = torch.from_numpy((py * g_test.T).T)
            model_f_dict['fc2.bias'] = torch.from_numpy(py).view(10)           
        model_ace_dict.update(model_f_dict)
        model_ace.load_state_dict(model_f_dict)

        if not torch.cuda.is_available():
            torch.save(model_nn.state_dict(), 'model_nn_cpu_pytorch_{}epochs.pkl'.format(nb_epochs))
            torch.save(model_ace.state_dict(), 'model_ace_cpu_pytorch_{}epochs.pkl'.format(nb_epochs))
    else:
        model_nn.load_state_dict(torch.load('model_nn_cpu_pytorch_1epochs.pkl'))
        model_ace.load_state_dict(torch.load('model_ace_cpu_pytorch_1epochs.pkl'))

    # # We use tf for evaluation on adversarial data
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_nn = convert_pytorch_model_to_tf(model_nn)
    tf_model_ace = convert_pytorch_model_to_tf(model_ace)
    cleverhans_model_nn = CallableModelWrapper(tf_model_nn, output_layer='logits')
    cleverhans_model_ace = CallableModelWrapper(tf_model_ace, output_layer='logits')

    # Create an FGSM attack
    for eps in np.arange(0.05,0.45,0.05):
        fgsm_params = {'eps': eps,
                       'clip_min': 0.,
                       'clip_max': 1.}

        fgsm_nn = FastGradientMethod(cleverhans_model_nn, sess=sess)
        adv_x_nn = fgsm_nn.generate(x_op, **fgsm_params)
        adv_pred_nn = tf_model_nn(adv_x_nn)

        fgsm_ace = FastGradientMethod(cleverhans_model_ace, sess=sess)
        adv_x_ace = fgsm_ace.generate(x_op, **fgsm_params)
        adv_pred_ace = tf_model_ace(adv_x_ace)
        # Run an evaluation of our model against fgsm
        total = 0
        correct_nn = 0
        correct_ace = 0
        for xs, ys in test_loader:
            adv_xs_nn, adv_preds_nn = sess.run([adv_x_nn, adv_pred_nn] , feed_dict={x_op: xs})
            adv_xs_ace, adv_preds_ace = sess.run([adv_x_ace, adv_pred_ace], feed_dict={x_op: xs})
            print(xs)
            # print(x_op[-1])
            # print(np.amax(adv_xs_nn[-1]),np.amin(adv_xs_nn[-1]))
            correct_nn += (np.argmax(adv_preds_nn, axis=1) == ys).sum()
            correct_ace += (np.argmax(adv_preds_ace, axis=1) == ys).sum()
            total += len(xs)

        acc_nn = float(correct_nn) / total
        acc_ace = float(correct_ace) / total
        print('eps:{}'.format(eps))
        print('nn Adv accuracy: {:.3f}'.format(acc_nn * 100))
        print('ace Adv accuracy: {:.3f}'.format(acc_ace * 100))
    # report.clean_train_adv_eval = acc
    # return report


def main(_=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
