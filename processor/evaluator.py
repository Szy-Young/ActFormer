import argparse
import yaml
import numpy as np
import h5py
from sklearn.metrics import confusion_matrix as cf
from scipy import linalg
import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
from utils import torchlight
from utils.torchlight import import_class, DictAction, str2bool

class Score_Processor:
    """
    Processor for Quantitative Evaluation
    """
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()

        self.load_model()
        self.load_data()


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


    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.model = self.io.load_weights(self.model, self.arg.weights).to(self.dev)


    def load_data(self):
        self.Feeder = import_class(self.arg.feeder)


    def start(self):
        if self.arg.sample is None:
            raise ValueError('Please appoint --sample.')
        if self.arg.data is None:
            raise ValueError('Please appoint --data.')
        if self.arg.acc:
            print('Calculating Acc. for generated samples...')
            self.compute_Acc()
        if self.arg.fid:
            print('Calculating FID for generated samples...')
            self.compute_FID()


    def organize_gen_data(self, sample):
        data_dict = {}
        with h5py.File(sample, 'r') as f:
            keys = list(f.keys())
            if len(f[keys[0]][:].shape) == 3:
                # the data is not saved in batch
                start_index = list(range(0, len(keys), self.arg.num_samples_per_class))
                count = 1
                for index in start_index:
                    batch_list = []
                    for i in range(index, index + self.arg.num_samples_per_class):
                        batch_list.append(f[keys[i]])
                    data_dict['A' + str(count).zfill(4)] = np.stack(batch_list, axis=0)
                    count += 1
            else:
                # the data is already saved in batch
                for key in keys:
                    data_dict[key] = f[key][:]
        return data_dict


    def compute_Acc(self):
        self.model.eval()

        for sample in self.arg.sample:
            label_list = []
            pred_list = []

            data_dict = self.organize_gen_data(sample)

            for k, v in tqdm.tqdm(data_dict.items()):
                labels = [int(k[1:]) - 1 for i in range(self.arg.num_samples_per_class)]
                label_list = label_list + labels
                data = torch.from_numpy(v).float().to(self.dev)
                data = data.unsqueeze(-1).permute(0, 1, 3, 2, 4).contiguous()
                with torch.no_grad():
                    output = self.model(data)
                    output = output.cpu().detach().numpy()
                    pred_list = pred_list + np.argmax(output, axis=1).tolist()

            confuse_matrix = cf(label_list, pred_list)
            acc = confuse_matrix.diagonal() / confuse_matrix.sum(axis=1)
            mean_acc = np.mean(acc)
            print('mean_accuracy: {}'.format(mean_acc))


    def compute_FID(self):
        # get mean and cov for real data
        print('Calculating activation statistics of true samples...')
        self.mt, self.st, self.nums, self.total_m, self.total_cov = self.calculate_activation_statistics(self.arg.data)

        # get mean and cov for generated data
        self.model.eval()
        for sample in self.arg.sample:
            act_means = np.zeros([self.arg.num_class, 256])
            act_covs = np.zeros([self.arg.num_class, 256, 256])
            fid_list, total_feature_list = [], []

            data_dict = self.organize_gen_data(sample)

            for k, v in tqdm.tqdm(data_dict.items()):
                data = torch.from_numpy(v).float().to(self.dev)
                data = data.unsqueeze(-1).permute(0, 1, 3, 2, 4).contiguous()

                if self.arg.aggregate:
                    # data_p_all = torch.split(data, 3, dim=1)
                    data_p_all = torch.split(data, 6, dim=1)
                    features = []
                    with torch.no_grad():
                        for data_p in data_p_all:
                            output, feature = self.model(data_p, True)
                            features.append(feature)
                        feature = torch.stack(features, 0).max(0)[0]
                else:
                    with torch.no_grad():
                        output, feature = self.model(data, True)

                feature = feature.cpu().detach().numpy().astype('double')
                total_feature_list.append(feature)
                act_means[int(k[1:]) - 1] = np.mean(feature, 0)
                act_covs[int(k[1:]) - 1] = np.cov(feature, rowvar=False)

            # eval FID for per class
            for i in range(self.arg.num_class):
                if self.nums[i] < 1:
                    continue
                fid = self.calculate_frechet_distance(self.mt[i], self.st[i], act_means[i], act_covs[i])
                fid_list.append(fid)

            # eval total FID and m_is
            total_feature = np.concatenate(total_feature_list, axis=0)
            total_act_mean = np.mean(total_feature, 0)
            total_act_cov = np.cov(total_feature, rowvar=False)
            total_fid = self.calculate_frechet_distance(self.total_m, self.total_cov, total_act_mean, total_act_cov)

            mean_fid = np.mean(fid_list)
            print('mean_fid: {}    whole_FID: {}'.format(mean_fid, total_fid))


    def get_activations(self, sample_path, frame_offset=-1):
        self.model.eval()
        loader = torch.utils.data.DataLoader(
            dataset=self.Feeder(
                path=sample_path,
                frame_offset=frame_offset,
                **self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=1)

        pred_arr, labels = [], []
        for data, label in loader:
            data = data.float().to(self.dev)

            if self.arg.aggregate:
                # data_p_all = torch.split(data, 3, dim=1)
                data_p_all = torch.split(data, 6, dim=1)
                outputs = []
                features = []
                with torch.no_grad():
                    for data_p in data_p_all:
                        output, feature = self.model(data_p, True)
                        outputs.append(output)
                        features.append(feature)
                    output = torch.stack(outputs, 0).max(0)[0]
                    feature = torch.stack(features, 0).max(0)[0]
                    pred_arr.append(feature.cpu().detach().numpy().astype('double'))
                    labels.append(label.cpu().detach().numpy())

            else:
                with torch.no_grad():
                    output, feature = self.model(data, True)
                    pred_arr.append(feature.cpu().detach().numpy().astype('double'))
                    labels.append(label.cpu().detach().numpy())

        return np.concatenate(pred_arr, 0), np.concatenate(labels, 0)


    def calculate_activation_statistics(self, sample_path, frame_offset=-1):
        act, labels = self.get_activations(sample_path, frame_offset)
        total_act_mean = np.mean(act, axis=0)
        total_act_cov = np.cov(act, rowvar=False)
        act_means = np.zeros([self.arg.num_class, 256])
        act_cov = np.zeros([self.arg.num_class, 256, 256])
        nums = np.zeros(self.arg.num_class)
        for i in range(self.arg.num_class):
            act_means[i] = np.mean(act[np.where(labels == i)[0]], axis=0)
            act_cov[i] = np.cov(act[np.where(labels == i)[0]], rowvar=False)
            nums[i] = len(act[np.where(labels == i)[0]])

        return act_means, act_cov, nums, total_act_mean, total_act_cov


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Evaluate generated results')

        # Basis
        parser.add_argument('--work_dir', default='work_dir/evaluator', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
        parser.add_argument('-s', '--sample', nargs='+')
        parser.add_argument('-d', '--data')
        parser.add_argument('--num_class', type=int, default=60)
        parser.add_argument('--num_samples_per_class', type=int, default=100)
        parser.add_argument('--acc', dest='acc', action='store_true')
        parser.add_argument('--fid', dest='fid', action='store_true')
        parser.add_argument('--aggregate', dest='aggregate', action='store_true')

        # Visualize and debug
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

        # Feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')

        # Model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')

        return parser