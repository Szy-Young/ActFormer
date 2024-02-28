import argparse
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix as cf

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
from utils import torchlight
from utils.torchlight import import_class, DictAction, str2bool

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()

        self.load_model()
        self.load_data()
        self.load_optimizer()
        self.fusion_model = False


    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                assert k in key, 'Unknown Arguments: {}'.format(k)
            parser.set_defaults(**default_arg)

        arg = parser.parse_args(argv)
        self.arg = arg


    def init_environment(self):
        self.io = torchlight.IO(self.arg.work_dir, save_log=self.arg.save_log, print_log=self.arg.print_log)
        self.io.save_arg(self.arg)
        self.dev = "cuda:0"
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)


    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args)).to(self.dev)
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if self.arg.resample:
            self.data_set, self.data_loader = dict(), dict()
            self.data_set['train'] = Feeder(**self.arg.train_feeder_args)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device))
        else:
            self.data_loader = dict()
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                drop_last=True)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device))


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.arg.base_lr)
        else:
            raise ValueError()


    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))


    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)
            self.io.print_log(info)


    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            coeff = 0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * coeff
            self.lr = coeff * self.arg.base_lr
        else:
            self.lr = self.arg.base_lr


    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        return accuracy


    def train(self):
        self.model.train()
        self.adjust_lr()

        if self.arg.resample:
            self.data_set['train'].resample()
            loader = torch.utils.data.DataLoader(
                dataset=self.data_set['train'],
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                drop_last=True)
        else:
            loader = self.data_loader['train']

        loss_value = []
        result = []
        labels = []

        for data, label in loader:
            # get data
            labels.append(label)
            data = data.unsqueeze(-1).permute(0, 1, 3, 2, 4) # [B, C, T, V, M]
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            result.append(output.data.cpu().numpy())
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.result = np.concatenate(result)
        self.label = np.concatenate(labels)
        acc = self.show_topk(1)
        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.epoch_info['mean_acc'] = acc
        self.epoch_info['best_acc'] = self.best_acc
        self.show_epoch_info()
        self.io.print_timer()


    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            # get data
            data = data.unsqueeze(-1).permute(0, 1, 3, 2, 4)
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
                if self.fusion_model:
                    output = output*(1-self.arg.fusion_rate) + self.fusion_model(data) * self.arg.fusion_rate
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()
            if self.arg.phase == 'train':
                pred = np.argmax(self.result, axis=1)
                confuse_matrix = cf(self.label, pred)
                acc = confuse_matrix.diagonal() / confuse_matrix.sum(axis=1)
                mean_acc = self.show_topk(1)

                if mean_acc > self.best_acc:
                    self.best_acc = mean_acc
                    self.best_acc_per_class = acc
                    self.best_cf = confuse_matrix
                    self.best_epoch = self.meta_info['epoch']
                    self.best_top5 = self.show_topk(5)
            # show top-k accuracy
            for k in self.arg.show_topk:
                accuracy = self.show_topk(k)
                self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))


    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.best_acc = 0
        self.best_epoch = -1

        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.train()
            self.io.print_log('Done.')

            # save model
            if ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch):
                filename = 'epoch{}_model.pt'.format(epoch + 1)
                self.io.save_model(self.model, filename)

            # evaluation
            if ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.test()
                self.io.print_log('Done.')
        print('Best acc---Top1: {} Top5: {} in epoch {}'.format(self.best_acc, self.best_top5, self.best_epoch))


    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Train a motion evaluator')

        # Basis
        parser.add_argument('--work_dir', default='work_dir/recognition', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
        parser.add_argument('--phase', type=str, default='train', help='train/eval')

        # Visualize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

        # Feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')

        # Model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')

        # Eval
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        # Train data sampling
        parser.add_argument('--resample', type=str2bool, default=False, help='square-root sampling or not')

        # Optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

        return parser
