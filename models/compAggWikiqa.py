
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nn.DMax import DMax
import util.loadFiles as tr
from util.utils import MAP, MRR


class CompAggWikiQA(nn.Module):
    def __init__(self, args):
        super(CompAggWikiQA, self).__init__()

        self.mem_dim = args.mem_dim
        self.cov_dim = args.cov_dim
        self.optim_state = { "learningRate": args.learning_rate }
        self.batch_size = args.batch_size
        self.emb_dim = args.wvecDim
        self.task = args.task
        self.numWords = args.numWords
        self.dropoutP = args.dropoutP
        self.comp_type = args.comp_type
        self.window_sizes = args.window_sizes
        self.window_large = args.window_large
        self.gpu = args.gpu

        self.best_score = 0

        self.emb_vecs = nn.Embedding(self.numWords, self.emb_dim)
        self.emb_vecs.weight.data = tr.loadVacab2Emb(self.task)

        self.ops = 'mul'

        self.criterion = nn.KLDivLoss()

        class Softmax(nn.Module):
            def __init__(self, mem_dim):
                super(Softmax, self).__init__()
                self.layer1 = nn.Linear(mem_dim, 1)

            def forward(self, input):
                var1 = self.layer1(input)
                var1 = var1.view(-1)
                out = F.log_softmax(var1, dim=0)
                return out

        self.soft_module = Softmax(self.mem_dim)

    def projection_layer(self, input):
        """
            Preprocessing step, main projection.
        """

        self.linear1 = nn.Linear(self.emb_dim, self.mem_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.mem_dim)

        i = nn.Sigmoid()(self.linear1(input))
        u = nn.Tanh()(self.linear2(input))
        out = i.mul(u)
        return out

    def attend(self, question, answer, withDropout=False):
        """
            Attention step, applying an Attention Layer.
        """
        question_projection = question
        if withDropout:
            question_projection = nn.Dropout(self.dropoutP).forward(question_projection)
        attention = torch.mm(question_projection, answer.t())
        question_weights = F.softmax(attention.transpose(0, 1), dim=0)
        context_vector = torch.mm(question_weights, question_projection)
        return context_vector

    def compare(self, answer, attention_vector):
        """
            Comparison step, multiplication works best for WikiQA.
        """
        if self.ops == "mul":
            out = answer.mul(attention_vector)
        return out

    def conv_module(self, input, sizes):
        """
            Aggregate step, Computes the convolutions along with dynamic max pooling.
        """
        conv = [None] * len(self.window_sizes)
        pool = [None] * len(self.window_sizes)
        input_view = input.view(1, input.size()[0], input.size()[1]).transpose(1, 2)
        for i, window_size in enumerate(self.window_sizes):
            tempconv = nn.Conv1d(self.cov_dim, self.mem_dim, window_size)(input_view)[0].transpose(0, 1)
            conv[i] = nn.ReLU()(tempconv)
            pool[i] = DMax(dimension=0, windowSize=window_size, gpu=self.gpu)(conv[i], sizes)

        concate = torch.cat(pool, 1)
        linear1 = nn.Linear(len(self.window_sizes) * self.mem_dim, self.mem_dim)(concate)
        output = nn.Tanh()(linear1)
        return output

    def context_encoding(self, q, a):
        """
            Context Encoding, applying the "projection layers" on both embeddings.
        """
        return self.projection_layer(q), self.projection_layer(a)

    def main_flow(self, data_q, data_as):
        """
            Main Computation of the model
        """
        data_as_len = torch.IntTensor(len(data_as))
        for k in range(len(data_as)):
            data_as_len[k] = data_as[k].size()[0]
            # Manually padding.
            if data_as_len[k] < self.window_large:
                as_tmp = torch.LongTensor(self.window_large).fill_(0)
                data_as[k] = as_tmp
                data_as_len[k] = self.window_large

        data_as_word = torch.cat(data_as, 0)
        inputs_a_emb = self.emb_vecs.forward(
            Variable(data_as_word.type(torch.LongTensor), requires_grad=False))
        inputs_q_emb = self.emb_vecs.forward(Variable(data_q, requires_grad=False))

        projs_q_emb, projs_a_emb = self.context_encoding(inputs_q_emb, inputs_a_emb)

        if data_q.size()[0] == 1:
            projs_q_emb = projs_q_emb.resize(1, self.mem_dim)

        att_output = self.attend(projs_q_emb, projs_a_emb)

        sim_output = self.compare(projs_a_emb, att_output)

        conv_output = self.conv_module(sim_output, data_as_len)

        soft_output = self.soft_module.forward(conv_output)

        return soft_output

    def forward(self, data_q, data_as):
        """
            Train model, Computation performed at every call.
        """
        output = self.main_flow(data_q, data_as)
        return output

    def predict(self, data_raw):
        """
            Prediction, Validation/Testing
        """
        data_q, data_as, label = data_raw
        output = self.main_flow(data_q, data_as)

        map = MAP(label, output.data)
        mrr = MRR(label, output.data)
        return map, mrr

    def predict_dataset(self, dataset):
        """
            Prediction on specified dataset (WikiQA: dev, prod, etc.)
        """
        self.emb_vecs.eval()
        res = [0., 0.]
        dataset_size = len(dataset)
        for i in range(dataset_size):
            prediction = self.predict(dataset[i])
            res[0] = res[0] + prediction[0]
            res[1] = res[1] + prediction[1]

        res[0] = res[0] / dataset_size
        res[1] = res[1] / dataset_size

        return res
