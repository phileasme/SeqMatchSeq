# Training settings
import argparse

import torch
from tqdm import tqdm
import util.loadFiles as tr
from torch.autograd import Variable
from models.compAggWikiqa import CompAggWikiQA
from util import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='number of sequences to train on in parallel')
parser.add_argument('--max_epochs', type=int, default=10, help='number of full passes through the training data')
parser.add_argument('--seed', type=int, default=321, help='torch manual random number generator seed')
parser.add_argument('--reg', type=float, default=0, help='regularize value')
parser.add_argument('--learning_rate', type=float, default=0.004, help='learning rate')
parser.add_argument('--emb_lr', type=float, default=0., help='embedding learning rate')
parser.add_argument('--emb_partial', type=bool, default=True, help='only update the non-pretrained embeddings')
parser.add_argument('--lr_decay', type=float, default=0.95, help='learning rate decay ratio')
parser.add_argument('--dropoutP', type=float, default=0.04, help='dropout ratio')
parser.add_argument('--expIdx', type=int, default=0, help='experiment index')
parser.add_argument('--num_classes', type=int, default=3, help='number of classes')

parser.add_argument('--wvecDim', type=int, default=300, help='embedding dimension')
parser.add_argument('--mem_dim', type=int, default=150, help='state dimension')

parser.add_argument('--cov_dim', type=int, default=150, help='conv dimension')
parser.add_argument('--window_sizes', type=list, default=[1, 2, 3, 4, 5], help='window sizes')
parser.add_argument('--window_large', type=int, default=5, help='largest window size')

parser.add_argument('--model', type=str, default="compAggWikiqa", help='model')
parser.add_argument('--task', type=str, default="wikiqa", help='task')

parser.add_argument('--comp_type', type=str, default="mul", help='w-by-w type')
parser.add_argument('--visualize', type=bool, default=False, help='visualize')

parser.add_argument('--preEmb', type=str, default="glove", help='Embedding pretrained method')
parser.add_argument('--grad', type=str, default="adamax", help='gradient descent method')

parser.add_argument('--log', type=str, default="nothing", help='log message')
parser.add_argument('--gpu', type=bool, default=False, help='use gpu or not')

opt = parser.parse_args()
torch.manual_seed(opt.seed)
torch.set_num_threads(1)

tr.init(opt)
vocab = tr.loadVocab(opt.task)
ivocab = tr.loadiVocab(opt.task)
opt.numWords = len(ivocab)
print("Vocal size: %s" % opt.numWords)
print('loading data ..')

train_dataset = tr.loadData('train', opt.task)
print("Size of training data: %s" % len(train_dataset))
dev_dataset = tr.loadData('dev', opt.task)
print("Size of dev data: %s" % len(dev_dataset))


def train(model: compAggWikiqa, dataset: list):
    model.emb_vecs.train()

    dataset_size = len(dataset)

    indices = [667] + [x for x in range(667)] + [x for x in range(668, 873)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in tqdm(range(0, dataset_size, model.batch_size)):
        batch_size = min(model.batch_size, dataset_size - i)

        loss = 0.
        for j in range(0, batch_size):
            idx = indices[i + j]
            data_raw = dataset[idx]
            data_q, data_as, label = data_raw
            label = Variable(label, requires_grad=False)
            soft_output = model(data_q, data_as)

            example_loss = model.criterion(soft_output, label)

            loss += example_loss
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    model = CompAggWikiQA(opt)
    for i in range(opt.max_epochs):
        train(model, train_dataset)
        model.optim_state['learningRate'] = model.optim_state['learningRate'] * opt.lr_decay

        recordDev = model.predict_dataset(dev_dataset)

        utils.save('./trainedmodel/', opt, [recordDev], i)
        # if i == opt.max_epochs then
        #     model.params:copy( model.best_params )
        #     recordDev = model:predict_dataset(dev_dataset)
        #     if opt.task == 'snli' or opt.task == 'wikiqa' then recordTest   = model:predict_dataset(test_dataset) end
        #     recordTrain  = model:predict_dataset(train_dataset)
        #     model.save('../trainedmodel/', opt, {recordDev, recordTest, recordTrain}, i)

if __name__ == "__main__":
    main()
