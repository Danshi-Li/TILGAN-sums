# Import comet_ml in the top of your file
from comet_ml import Experiment
from comet_ml import Optimizer
import argparse
import os
import time
import math
import numpy as np
import random
import sys
import shutil
import json
import string
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl, create_exp_dir
from models import Seq2Seq, MLP_D, MLP_D_local, MLP_G
from bleu_self import *
from bleu_test import *
#import wandb
#wandb.init(project="t-arae-lang_cal_param")
import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# We only need to specify the algorithm and hyperparameters to use:

# Declare your hyperparameters search:
'''
config = {
    "algorithm": "bayes",
    "parameters": {
        # "size": {"type": "integer", "min": 1, "max": 10},
        # "lr": {"type": "float", "min": 0.1, "max": 1.0},
        "aehidden": {"type": "integer", "min": 10, "max": 512},
        "nlayers": {"type": "integer", "min": 1, "max": 6},
        "niters_ae": {"type": "integer", "min": 1, "max": 6},
        "niters_gan_d": {"type": "integer", "min": 1, "max": 6},
        "niters_gan_g": {"type": "integer", "min": 1, "max": 6},
        "niters_gan_ae": {"type": "integer", "min": 1, "max": 6},
        "lr_ae": {"type": "float", "min": 0.01, "max": 10},
        "lr_gan_e": {"type": "float", "min": 1e-05, "max": 1e-02},
        "lr_gan_g": {"type": "float", "min": 1e-05, "max": 1e-02},
        "lr_gan_d": {"type": "float", "min": 1e-05, "max": 1e-02},
        "clip": {"type": "float", "min": 0.2, "max": 5},
        "z_size": {"type": "integer", "min": 50, "max": 512},
    },
    "spec": {
        "randomize": True,
        "metric": "testbleu4",
        "objective": "maximize",
    },
}
'''


config = {"algorithm": "grid",
 "spec": {
    "randomize": False,
    "maxCombo": 0,
    "metric": "testbleu4",
    "objective": "maximize",
    # "seed": 42,
    "gridSize": 10,
    "minSampleSize": 100,
    "retryLimit": 20,
    "retryAssignLimit": 0,
 },
 "trials": 1,
 "parameters": {
                # "aehidden": {"type": "discrete", "values": [28, 56, 128, 256]},
                # "nlayers": {"type": "discrete", "values": [2,3,4]},
                # "niters_ae": {"type": "discrete", "values": [1, 2, 3]},
                # "niters_gan_d": {"type": "discrete", "values": [1, 2, 3]},
                # "niters_gan_g": {"type": "discrete", "values": [1, 2, 3]},
                # "niters_gan_ae": {"type": "discrete", "values": [1, 2, 3]},
                # "lr_ae": {"type": "discrete", "values": [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]},
                # "lr_gan_e": {"type": "discrete", "values": [1e-05, 3e-05, 5e-05, 1e-04, 4e-04, 8e-04]},
                # "lr_gan_g": {"type": "discrete", "values": [1e-05, 3e-05, 5e-05, 1e-04, 4e-04, 8e-04]},
                # "lr_gan_d": {"type": "discrete", "values": [1e-05, 3e-05, 5e-05, 1e-04, 4e-04, 8e-04]},
                # "clip": {"type": "discrete", "values": [1, 3, 5]},
                # "z_size": {"type": "discrete", "values": [100, 256, 512]},
                "aehidden": {"type": "discrete", "values": [56]},
                "nlayers": {"type": "discrete", "values": [2]},
                "niters_ae": {"type": "discrete", "values": [1]},
                "niters_gan_d": {"type": "discrete", "values": [2]},
                "niters_gan_dec": {"type": "discrete", "values": [1]},
                "niters_gan_g": {"type": "discrete", "values": [1]},
                "niters_gan_ae": {"type": "discrete", "values": [2]},
                # "lr_ae": {"type": "discrete", "values": [0.5]},
                "lr_ae": {"type": "discrete", "values": [0.001]},
                "lr_gan_e": {"type": "discrete", "values": [1e-04]},
                "lr_gan_g": {"type": "discrete", "values": [5e-05]},
                "lr_gan_d": {"type": "discrete", "values": [1e-04]},
                "clip": {"type": "discrete", "values": [1]},
                },
 "name": "My Grid Search Optimizer",
}


# Next, create an optimizer, passing in the config:
# (You can leave out API_KEY if you already set it)
#NOTE ENTER YOUR API_KEY HERE
comet_opt = Optimizer(config, api_key="", project_name="", workspace="")
for experiment in comet_opt.get_experiments():

    '''
    code for debug
    '''
    import argparse

    parser = argparse.ArgumentParser(description='ARAE for SNLI')
    # Path Arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--kenlm_path', type=str, default='./kenlm',
                        help='path to kenlm directory')
    parser.add_argument('--save', type=str, default='snli_example',
                        help='output directory name')

    # Data Processing Arguments
    parser.add_argument('--maxlen', type=int, default=15,
                        help='maximum length')
    parser.add_argument('--vocab_size', type=int, default=0,
                        help='cut vocabulary down to this size '
                             '(most frequently seen words in train)')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                        help='lowercase all text')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                        help='not lowercase all text')
    parser.set_defaults(lowercase=True)

    # Model Arguments
    parser.add_argument('--emsize', type=int, default=512,
                        help='size of word embeddings')
    parser.add_argument('--nhidden', type=int, default=512,
                        help='number of hidden units per layer')
    # parser.add_argument('--nlatent', type=int, default=8192,
    #                     help='number of hidden units per layer for discriminator and decoder (because we use concatenation, so it is len*nhidden)')
    parser.add_argument('--nlayers', type=int, default=experiment.get_parameter("nlayers"),
                        help='number of layers')
    parser.add_argument('--noise_r', type=float, default=0.05,
                        help='stdev of noise for autoencoder (regularizer)')
    parser.add_argument('--nheads', type=int, default=4,
                        help='number of heads')
    parser.add_argument('--nff', type=int, default=1024,
                        help='feedforward network dimension in Transformer')
    parser.add_argument('--aehidden', type=int, default=experiment.get_parameter("aehidden"),
                        help='the squeezed hidden dimension')
    parser.add_argument('--noise_anneal', type=float, default=0.9995,
                        help='anneal noise_r exponentially by this'
                             'every 100 iterations')
    parser.add_argument('--hidden_init', action='store_true',
                        help="initialize decoder hidden state with encoder's")
    parser.add_argument('--arch_g', type=str, default='300-300',
                        help='generator architecture (MLP)')
    parser.add_argument('--arch_d', type=str, default='300-300',
                        help='critic/discriminator architecture (MLP)')
    parser.add_argument('--arch_d_local', type=str, default='300-300',
                        help='local critic/discriminator architecture (MLP)')
    parser.add_argument('--z_size', type=int, default=100,
                        help='dimension of random noise z to feed into generator')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--noise_seq_length', type=int, default=15,
                        help='the seq_length of fake noise ')
    parser.add_argument('--gan_type', type=str, default='kl', choices=['kl', 'all', 'wgan'],
                        help='generator architecture (MLP)')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='maximum number of epochs')
    parser.add_argument('--min_epochs', type=int, default=12,
                        help="minimum number of epochs to train for")
    parser.add_argument('--no_earlystopping', action='store_true',
                        help="won't use KenLM for early stopping")
    parser.add_argument('--patience', type=int, default=2,
                        help="number of language model evaluations without ppl "
                             "improvement to wait before early stopping")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='eval batch size')
    parser.add_argument('--niters_ae', type=int, default=experiment.get_parameter("niters_ae"),
                        help='number of autoencoder iterations in training')
    parser.add_argument('--niters_gan_d', type=int, default=experiment.get_parameter("niters_gan_d"),
                        help='number of discriminator iterations in training')
    parser.add_argument('--niters_gan_dec', type=int, default=experiment.get_parameter("niters_gan_dec"),
                        help='number of enhance decoder')
    parser.add_argument('--niters_gan_g', type=int, default=experiment.get_parameter("niters_gan_g"),
                        help='number of generator iterations in training')
    parser.add_argument('--niters_gan_ae', type=int, default=experiment.get_parameter("niters_gan_ae"),
                        help='number of gan-into-ae iterations in training')
    parser.add_argument('--niters_gan_schedule', type=str, default='',
                        help='epoch counts to increase number of GAN training '
                             ' iterations (increment by 1 each time)')
    parser.add_argument('--lr_ae', type=float, default=experiment.get_parameter("lr_ae"),
                        help='autoencoder learning rate')
    parser.add_argument('--lr_gan_e', type=float, default=experiment.get_parameter("lr_gan_e"),
                        help='gan encoder learning rate')
    parser.add_argument('--lr_gan_g', type=float, default=experiment.get_parameter("lr_gan_g"),
                        help='generator learning rate')
    parser.add_argument('--lr_gan_d', type=float, default=experiment.get_parameter("lr_gan_d"),
                        help='critic/discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--clip', type=float, default=experiment.get_parameter("clip"),
                        help='gradient clipping, max norm')
    parser.add_argument('--gan_clamp', type=float, default=0.01,
                        help='WGAN clamp')
    parser.add_argument('--gan_gp_lambda', type=float, default=1,
                        help='WGAN GP penalty lambda')
    parser.add_argument('--gan_lambda', type=float, default=0.1,
                        help='coefficient of divergence (minimized with GAN)')
    parser.add_argument('--add_noise', action='store_true',
                        help='whether to add_noise, default is False')
    parser.add_argument('--gan_d_local', action='store_true',
                        help='whether to turn on gan_d_local, default is False')
    parser.add_argument('--gan_d_local_windowsize', type=int, default=3,
                        help='gan_d_local_windowsize')
    parser.add_argument('--gan_g_activation', action='store_true',
                        help='whether to turn on activation of gan_g, default is False')
    parser.add_argument('--enhance_dec', action='store_true',
                        help='whether to enhance decoder')

    # Evaluation Arguments
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--N', type=int, default=5,
                        help='N-gram order for training n-gram language model')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='interval to log autoencoder training results')

    # Other
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    args = parser.parse_args()
    print(vars(args))
    args.save = args.save+now_time
    # Report any information you need by:
    hyper_params = vars(args)
    #wandb.config.update(args)  # adds all of the arguments as config variables
    experiment.log_parameters(hyper_params)


    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    # create corpus
    corpus = Corpus(args.data_path,
                    maxlen=args.maxlen,
                    vocab_size=args.vocab_size,
                    lowercase=args.lowercase)

    # save arguments
    ntokens = len(corpus.dictionary.word2idx)
    print("Vocabulary Size: {}".format(ntokens))
    args.ntokens = ntokens

    # exp dir
    create_exp_dir(os.path.join(args.save), ['train.py', 'models.py', 'utils.py'],
            dict=corpus.dictionary.word2idx, options=args)

    def logging(str, to_stdout=True):
        with open(os.path.join(args.save, 'log.txt'), 'a') as f:
            f.write(str + '\n')
        if to_stdout:
            print(str)
    logging(str(vars(args)))

    # eval_batch_size = 10
    eval_batch_size = args.eval_batch_size
    noise_seq_length = args.noise_seq_length
    test_data = batchify(corpus.test, eval_batch_size, args.maxlen, shuffle=False)
    train_data = batchify(corpus.train, args.batch_size, args.maxlen,  shuffle=True)

    print("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################
    autoencoder = Seq2Seq(add_noise=args.add_noise,
                          emsize=args.emsize,
                          nhidden=args.nhidden,
                          ntokens=args.ntokens,
                          nlayers=args.nlayers,
                          nheads=args.nheads,
                          nff=args.nff,
                          aehidden=args.aehidden,
                          noise_r=args.noise_r,
                          hidden_init=args.hidden_init,
                          dropout=args.dropout,
                          gpu=True)
    nlatent = args.aehidden * (args.maxlen+1)
    gan_gen = MLP_G(ninput=args.z_size, noutput=nlatent, layers=args.arch_g, gan_g_activation=args.gan_g_activation)
    gan_disc = MLP_D(ninput=nlatent, noutput=1, layers=args.arch_d)
    gan_disc_local = MLP_D_local(ninput=args.gan_d_local_windowsize * args.aehidden, noutput=1, layers=args.arch_d_local)
    #wandb.watch(autoencoder)
    #wandb.watch(gan_gen)
    #wandb.watch(gan_disc)
    #wandb.watch(gan_disc_local)

    optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
    # scheduler_ae = torch.optim.lr_scheduler.StepLR(optimizer_ae, 4, gamma=0.95)
    # optimizer_gan_e = optim.SGD(autoencoder.encoder.parameters(), lr=args.lr_ae)

    # not sure whether just ".encoder"
    optimizer_gan_e = optim.Adam(autoencoder.encoder.parameters(),
                                 lr=args.lr_gan_e,
                                 betas=(args.beta1, 0.999))
    optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                                 lr=args.lr_gan_g,
                                 betas=(args.beta1, 0.999))
    optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                                 lr=args.lr_gan_d,
                                 betas=(args.beta1, 0.999))
    optimizer_gan_d_local = optim.Adam(gan_disc_local.parameters(),
                                 lr=args.lr_gan_d,
                                 betas=(args.beta1, 0.999))
    optimizer_gan_dec = optim.Adam(autoencoder.decoder.parameters(),
                                 lr=args.lr_gan_e,
                                 betas=(args.beta1, 0.999))

    autoencoder = autoencoder.to(device)
    gan_gen = gan_gen.to(device)
    gan_disc = gan_disc.to(device)
    gan_disc_local = gan_disc_local.to(device)

    ###############################################################################
    # Training code
    ###############################################################################
    def save_model():
        print("Saving models to {}".format(args.save))
        torch.save({
            "ae": autoencoder.state_dict(),
            "gan_g": gan_gen.state_dict(),
            "gan_d": gan_disc.state_dict(),
            "gan_d_local": gan_disc_local.state_dict()

            },
            os.path.join(args.save, "model.pt"))

    def cal_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm


        # for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        #     print(p.grad.data.norm(2).item())

    def load_models():
        model_args = json.load(open(os.path.join(args.save, 'options.json'), 'r'))
        word2idx = json.load(open(os.path.join(args.save, 'vocab.json'), 'r'))
        idx2word = {v: k for k, v in word2idx.items()}

        print('Loading models from {}'.format(args.save))
        loaded = torch.load(os.path.join(args.save, "model.pt"))
        autoencoder.load_state_dict(loaded.get('ae'))
        gan_gen.load_state_dict(loaded.get('gan_g'))
        gan_disc.load_state_dict(loaded.get('gan_d'))
        gan_disc_local.load_state_dict(loaded.get('gan_d_local'))
        return model_args, idx2word, autoencoder, gan_gen, gan_disc

    def evaluate_autoencoder(data_source, epoch):
        # Turn on evaluation mode which disables dropout.
        autoencoder.eval()
        # autoencoder.train()
        total_loss = 0
        ntokens = len(corpus.dictionary.word2idx)
        all_accuracies = 0
        bcnt = 0
        for i, batch in enumerate(data_source):
            source, target, lengths = batch
            # source = Variable(source.cuda(), volatile=True)
            # target = Variable(target.cuda(), volatile=True)
            with torch.no_grad():
                source = Variable(source.to(device))
                target = Variable(target.to(device))
                mask = target.gt(0)
                masked_target = target.masked_select(mask)
                # examples x ntokens
                output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

                # output: batch x seq_len x ntokens
                output = autoencoder(source, lengths, source, add_noise=args.add_noise, soft=False)
                flattened_output = output.view(-1, ntokens)

                masked_output = \
                    flattened_output.masked_select(output_mask).view(-1, ntokens)
                # total_loss += F.cross_entropy(masked_output, masked_target).data
                total_loss += F.cross_entropy(masked_output, masked_target)

                # accuracy
                # probs = F.softmax(masked_output, dim=-1) # added by diao
                max_vals, max_indices = torch.max(masked_output, 1)
                # print("----evaluate_AE----")
                # print("max_indices: ", max_indices)
                # print("masked_target: ", masked_target)
                # print("----evaluate_AE----")
                accuracy = torch.mean(max_indices.eq(masked_target).float()).data.item()
                # print("flag: accu = ", accuracy)
                all_accuracies += accuracy
                # all_accuracies += \
                #     torch.mean(max_indices.eq(masked_target).float()).data.item()
                bcnt += 1

            aeoutf = os.path.join(args.save, "autoencoder.txt")
            with open(aeoutf, "w") as f:
                max_values, max_indices = torch.max(output, 2)
                max_indices = \
                    max_indices.view(output.size(0), -1).data.cpu().numpy()
                target = target.view(output.size(0), -1).data.cpu().numpy()
                for t, idx in zip(target, max_indices):
                    # real sentence
                    chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                    f.write(chars + '\n')
                    # autoencoder output sentence
                    chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                    f.write(chars + '\n'*2)

        return total_loss.item() / len(data_source), all_accuracies/bcnt


    def gen_fixed_noise(noise, to_save):
        #noise [64,100] fake_hidden [64,100*max_len]
        gan_gen.eval()
        autoencoder.eval()

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)

        with open(to_save, "w") as f:
            max_indices = max_indices.data.cpu().numpy()
            for idx in max_indices:
                # generated sentence
                words = [corpus.dictionary.idx2word[x] for x in idx]
                # truncate sentences to first occurrence of <eos>
                truncated_sent = []
                for w in words:
                    if w != '<eos>':
                        truncated_sent.append(w)
                    else:
                        break
                chars = " ".join(truncated_sent)
                f.write(chars + '\n')

    def eval_bleu(gen_text_savepath):
        selfbleu = bleu_self(gen_text_savepath)
        real_text = os.path.join(args.data_path, "test.txt")
        testbleu = bleu_test(real_text, gen_text_savepath)
        return selfbleu, testbleu

    def train_lm(data_path):
        save_path = os.path.join("/tmp", ''.join(random.choice(
                string.ascii_uppercase + string.digits) for _ in range(6)))

        indices = []
        noise = Variable(torch.ones(100, args.z_size).to(device))
        for i in range(1000):
            # print("i:", i)
            noise.data.normal_(0, 1)
            fake_hidden = gan_gen(noise)
            max_indices = autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)
            indices.append(max_indices.data.cpu().numpy())
        indices = np.concatenate(indices, axis=0)

        with open(save_path, "w") as f:
            # laplacian smoothing
            for word in corpus.dictionary.word2idx.keys():
                f.write(word+'\n')
            for idx in indices:
                words = [corpus.dictionary.idx2word[x] for x in idx]
                # truncate sentences to first occurrence of <eos>
                truncated_sent = []
                for w in words:
                    if w != '<eos>':
                        truncated_sent.append(w)
                    else:
                        break
                chars = " ".join(truncated_sent)
                f.write(chars+'\n')
        # reverse ppl
        try:
            rev_lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                                data_path=save_path,
                                output_path=save_path+".arpa",
                                N=args.N)
            with open(os.path.join(args.data_path, 'test.txt'), 'r') as f:
                lines = f.readlines()
            if args.lowercase:
                lines = list(map(lambda x: x.lower(), lines))
            sentences = [l.replace('\n', '') for l in lines]
            rev_ppl = get_ppl(rev_lm, sentences)
        except:
            print("reverse ppl error: it maybe the generated files aren't valid to obtain an LM")
            rev_ppl = 1e15
        # forward ppl
        for_lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                            data_path=os.path.join(args.data_path, 'train.txt'),
                            output_path=save_path+".arpa",
                            N=args.N)
        with open(save_path, 'r') as f:
            lines = f.readlines()
        sentences = [l.replace('\n', '') for l in lines]
        for_ppl = get_ppl(for_lm, sentences)
        return rev_ppl, for_ppl


    def train_ae(epoch, batch, total_loss_ae, start_time, i):
        '''Train AE with the negative log-likelihood loss'''
        autoencoder.train()
        optimizer_ae.zero_grad()

        '''
        source:  [64,15] = [batchsize, max_len] 全是idx，不是embedding
        target: [960] = [batchsize*max_len]
        '''
        source, target, lengths = batch
        source = Variable(source.to(device))  #cuda:0 torch.int64
        target = Variable(target.to(device))  #cuda:0 torch.int64
        output = autoencoder(source, lengths, source, add_noise=args.add_noise, soft=False)  #output [15,64,512] [max_len, batchsize, n_hidden]

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
        flat_output = output.view(-1, ntokens)
        masked_output = flat_output.masked_select(output_mask).view(-1, ntokens)
        loss = F.cross_entropy(masked_output, masked_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
        # total_norm = torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
        train_ae_norm = cal_norm(autoencoder)
        # logging("train_ae_norm = " + str(train_ae_norm), to_stdout=False)
        optimizer_ae.step()

        # total_loss_ae += loss.data[0]
        total_loss_ae += loss.data.item()
        if i % args.log_interval == 0:
            probs = F.softmax(masked_output, dim=-1)
            max_vals, max_indices = torch.max(probs, 1)
            # print("----train_AE----")
            # print("max_indices: ", max_indices)
            # print("masked_target: ", masked_target)
            # print("----train_AE----")
            accuracy = torch.mean(max_indices.eq(masked_target).float()).data.item()
            cur_loss = total_loss_ae / args.log_interval
            elapsed = time.time() - start_time
            # logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:08.6f} | ms/batch {:5.2f} | '
            #         'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f} | train_ae_norm {:8.2f}'.format(
            #         epoch, i, len(train_data), scheduler_ae.get_lr()[0],
            #         elapsed * 1000 / args.log_interval,
            #         cur_loss, math.exp(cur_loss), accuracy, train_ae_norm))
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:08.6f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f} | train_ae_norm {:8.2f}'.format(
                    epoch, i, len(train_data), 0,
                    elapsed * 1000 / args.log_interval,
                    cur_loss, math.exp(cur_loss), accuracy, train_ae_norm))
            experiment.log_metric("loss", cur_loss)
            experiment.log_metric("ppl", math.exp(cur_loss))
            experiment.log_metric("acc", accuracy)
            experiment.log_metric("train_ae_total_norm", train_ae_norm)
            #wandb.log({"loss": cur_loss, "ppl": math.exp(cur_loss), "acc": accuracy, "train_ae_total_norm": train_ae_norm})

            total_loss_ae = 0
            start_time = time.time()
        return total_loss_ae, start_time


    def train_gan_g(gan_type='kl'):
        gan_gen.train()
        optimizer_gan_g.zero_grad()

        z = Variable(torch.Tensor(args.batch_size, args.z_size).normal_(0, 1).to(device))  #[64,100]
        fake_hidden = gan_gen(z)  #[64,3300]
        # fake_hidden.register_hook(grad_hook)
        fake_score = gan_disc(fake_hidden)

        if args.gan_d_local:
            idx = random.randint(0, args.maxlen - args.gan_d_local_windowsize)
            fake_hidden_local = fake_hidden[:, idx * args.aehidden : (idx + args.gan_d_local_windowsize) * args.aehidden]
            fake_score_local = gan_disc_local(fake_hidden_local)

            if gan_type == 'kl':
                errG = -(torch.exp(fake_score.detach()).clamp(0.5, 2) * fake_score).mean() -(torch.exp(fake_score_local.detach()).clamp(0.5, 2) * fake_score_local).mean()
            else:  # all or wgan
                errG = -fake_score.mean() -fake_score_local.mean()
        else:
            if gan_type == 'kl':
                errG = -(torch.exp(fake_score.detach()).clamp(0.5, 2) * fake_score).mean()
            else: # all or wgan
                errG = -fake_score.mean()


        errG *= args.gan_lambda
        errG.backward()

        train_gan_g_norm = cal_norm(gan_gen)
        # logging("train_gan_g_norm = " + str(train_gan_g_norm), to_stdout=False)
        experiment.log_metric("train_gan_g_norm", train_gan_g_norm)

        optimizer_gan_g.step()

        return errG


    def train_gan_dec(gan_type='kl'):
        autoencoder.decoder.train()
        optimizer_gan_dec.zero_grad()

        z = Variable(torch.Tensor(args.batch_size, args.z_size).normal_(0, 1).to(device))  #[64,100]
        fake_hidden = gan_gen(z)  #[64,1848]    1848 = 56*33  aehidden*(maxlen+1)
        # fake_hidden.register_hook(grad_hook)

        # 1. decoder  - soft distribution
        enhance_source, max_indices= autoencoder.generate_enh_dec(fake_hidden, args.maxlen, sample=args.sample)   #enhance_source: [64, 32, 3455]torch.float32   max_indices: [64, 32] torch.int64
        # 2. soft distribution - > encoder  -> fake_hidden
        enhance_hidden = autoencoder(enhance_source, None, max_indices, add_noise=args.add_noise, soft=True, encode_only=True)
        #enhance_hidden: [64, 1792]    1792 = 56*32 aehidden * maxlen
        fake_score = gan_disc(enhance_hidden)

        if args.gan_d_local:
            idx = random.randint(0, args.maxlen - args.gan_d_local_windowsize)
            fake_hidden_local = fake_hidden[:, idx * args.aehidden : (idx + args.gan_d_local_windowsize) * args.aehidden]
            fake_score_local = gan_disc_local(fake_hidden_local)

            if gan_type == 'kl':
                errG = -(torch.exp(fake_score.detach()).clamp(0.5, 2) * fake_score).mean() -(torch.exp(fake_score_local.detach()).clamp(0.5, 2) * fake_score_local).mean()
            else:  # all or wgan
                errG = -fake_score.mean() -fake_score_local.mean()
        else:
            if gan_type == 'kl':
                errG = -(torch.exp(fake_score.detach()).clamp(0.5, 2) * fake_score).mean()
            else: # all or wgan
                errG = -fake_score.mean()


        errG *= args.gan_lambda
        errG.backward()

        train_gan_g_norm = cal_norm(gan_gen)
        # logging("train_gan_g_norm = " + str(train_gan_g_norm), to_stdout=False)
        experiment.log_metric("train_gan_g_norm", train_gan_g_norm)

        # optimizer_gan_g.step()
        optimizer_gan_dec.step()

        return errG

    def grad_hook(grad):
        #gan_norm = torch.norm(grad, p=2, dim=1).detach().data.mean()
        #print(gan_norm, autoencoder.grad_norm)
        return grad * args.gan_lambda


    ''' Steal from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py '''
    def calc_gradient_penalty(netD, real_data, fake_data):
        bsz = real_data.size(0)
        alpha = torch.rand(bsz, 1)
        alpha = alpha.expand(bsz, real_data.size(1))  # only works for 2D XXX
        alpha = alpha.to(device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gan_gp_lambda
        return gradient_penalty


    def train_gan_d(batch, gan_type='kl'):
        gan_disc.train()
        gan_disc_local.train()
        optimizer_gan_d.zero_grad()
        optimizer_gan_d_local.zero_grad()

        # + samples
        source, target, lengths = batch
        source = Variable(source.to(device))  #[64, 33]
        target = Variable(target.to(device))  #[2112] 2112 = 64*33
        real_hidden = autoencoder(source, lengths, source, add_noise=args.add_noise, soft=False, encode_only=True)
        real_score = gan_disc(real_hidden.detach())

        idx = random.randint(0, args.maxlen - args.gan_d_local_windowsize)
        if args.gan_d_local:
            real_hidden_local = real_hidden[:, idx * args.aehidden : (idx + args.gan_d_local_windowsize) * args.aehidden]
            real_score_local = gan_disc_local(real_hidden_local)
            real_score += real_score_local


        if gan_type == 'wgan':
            errD_real = -real_score.mean()
        else: # kl or all
            errD_real = F.softplus(-real_score).mean()
        errD_real.backward()

        # - samples
        z = Variable(torch.Tensor(args.batch_size, args.z_size).normal_(0, 1).to(device))
        fake_hidden = gan_gen(z)
        fake_score = gan_disc(fake_hidden.detach())

        if args.gan_d_local:
            # idx = random.randint(0, args.maxlen)
            fake_hidden_local = fake_hidden[:, idx * args.aehidden : (idx + args.gan_d_local_windowsize) * args.aehidden]
            fake_score_local = gan_disc_local(fake_hidden_local)
            fake_score += fake_score_local

        if gan_type == 'wgan':
            errD_fake = fake_score.mean()
        else:  # kl or all
            errD_fake = F.softplus(fake_score).mean()
        errD_fake.backward()

        # gradient penalty
        if gan_type == 'wgan':
            gradient_penalty = calc_gradient_penalty(gan_disc, real_hidden.data, fake_hidden.data)
            gradient_penalty.backward()

        train_gan_d_norm = cal_norm(gan_disc)
        # logging("train_gan_d_norm = " + str(train_gan_d_norm), to_stdout=False)
        experiment.log_metric("train_gan_d_norm", train_gan_d_norm)

        optimizer_gan_d.step()
        optimizer_gan_d_local.step()
        return errD_real + errD_fake, errD_real, errD_fake


    def train_gan_d_into_ae(batch):
        autoencoder.train()
        optimizer_gan_e.zero_grad()

        source, target, lengths = batch
        source = Variable(source.to(device))
        target = Variable(target.to(device))
        real_hidden = autoencoder(source, lengths, source, add_noise=args.add_noise, soft=False, encode_only=True)
        # real_hidden.register_hook(grad_hook)

        if args.gan_d_local:
            idx = random.randint(0, args.maxlen - args.gan_d_local_windowsize)
            real_hidden_local = real_hidden[:, idx * args.aehidden : (idx + args.gan_d_local_windowsize) * args.aehidden]
            real_score_local = gan_disc_local(real_hidden_local)
            errD_real = gan_disc(real_hidden).mean() + real_score_local.mean()
        else:
            errD_real = gan_disc(real_hidden).mean()

        errD_real *= args.gan_lambda
        errD_real.backward()
        torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

        gan_d_into_ae_norm = cal_norm(autoencoder)
        # logging("gan_d_into_ae_norm = " + str(gan_d_into_ae_norm), to_stdout=False)
        experiment.log_metric("gan_d_into_ae_norm", gan_d_into_ae_norm)

        optimizer_gan_e.step()
        return errD_real


    def train():
        logging("Training")
        train_data = batchify(corpus.train, args.batch_size, args.maxlen, shuffle=True)

        # gan: preparation
        if args.niters_gan_schedule != "":
            gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
        else:
            gan_schedule = []
        niter_gan = 1
        # fixed_noise = Variable(torch.ones(args.batch_size, args.z_size).normal_(0, 1).cuda())
        # fixed_noise = Variable(torch.ones(args.batch_size, args.z_size).normal_(0, 1).cuda())
        fixed_noise = Variable(torch.ones(args.eval_batch_size, args.z_size).normal_(0, 1).to(device))

        best_rev_ppl = None
        impatience = 0
        for epoch in range(1, args.epochs+1):
            # update gan training schedule
            if epoch in gan_schedule:
                niter_gan += 1
                logging("GAN training loop schedule: {}".format(niter_gan))

            total_loss_ae = 0
            epoch_start_time = time.time()
            start_time = time.time()
            niter = 0
            niter_g = 1

            while niter < len(train_data):
                # train ae
                for i in range(args.niters_ae):
                    if niter >= len(train_data):
                        break  # end of epoch
                    total_loss_ae, start_time = train_ae(epoch, train_data[niter],
                                    total_loss_ae, start_time, niter)
                    niter += 1
                # train gan
                for k in range(niter_gan):
                    for i in range(args.niters_gan_d):
                        errD, errD_real, errD_fake = train_gan_d(
                                train_data[random.randint(0, len(train_data)-1)], args.gan_type)
                    for i in range(args.niters_gan_ae):
                        train_gan_d_into_ae(train_data[random.randint(0, len(train_data)-1)])
                    for i in range(args.niters_gan_g):
                        errG = train_gan_g(args.gan_type)
                    if args.enhance_dec:
                        for i in range(args.niters_gan_dec):
                            errG_enh_dec = train_gan_dec()
                    else:
                        errG_enh_dec = torch.Tensor([0])

                niter_g += 1
                if niter_g % 200 == 0:
                    # autoencoder.noise_anneal(args.noise_anneal)
                    logging('[{}/{}][{}/{}] Loss_D: {:.8f} (Loss_D_real: {:.8f} '
                            'Loss_D_fake: {:.8f}) Loss_G: {:.8f} Loss_Enh_Dec: {:.8f}'.format(
                             epoch, args.epochs, niter, len(train_data),
                             errD.data.item(), errD_real.data.item(),
                             errD_fake.data.item(), errG.data.item(), errG_enh_dec.data.item()))
                    experiment.log_metric("Loss_D", errD.data.item())
                    experiment.log_metric("Loss_D_real", errD_real.data.item())
                    experiment.log_metric("Loss_D_fake", errD_fake.data.item())
                    experiment.log_metric("Loss_G", errG.data.item())
                    experiment.log_metric("Loss_enh_dec", errG_enh_dec.data.item())
                    #wandb.log({"Loss_D": errD.data.item(), "Loss_D_real": errD_real.data.item(), "Loss_D_fake": errD_fake.data.item(), "Loss_G": errG.data.item(), "Loss_enh_dec":errG_enh_dec.data.item()})
            # scheduler_ae.step()
            # eval
            test_loss, accuracy = evaluate_autoencoder(test_data, epoch)
            logging('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                    'test ppl {:5.2f} | acc {:3.3f}'.format(epoch,
                    (time.time() - epoch_start_time), test_loss,
                    math.exp(test_loss), accuracy))
            experiment.log_metric("test loss", test_loss)
            experiment.log_metric("test ppl", math.exp(test_loss))
            experiment.log_metric("test acc", accuracy)
            #wandb.log({"test loss": test_loss, "test ppl": math.exp(test_loss), "test acc": accuracy})

            gen_text_savepath = os.path.join(args.save, "{:03d}_examplar_gen".format(epoch))
            gen_fixed_noise(fixed_noise, gen_text_savepath)
            if epoch % 5 == 0 or epoch % 4 == 0 or (args.epochs - epoch) <=2:
                #Due to the huge time cost, we only evaluate the bleu for last 3 epochs
                selfbleu, testbleu = eval_bleu(gen_text_savepath)
                logging('bleu_self: [{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}]'.format(selfbleu[0], selfbleu[1], selfbleu[2], selfbleu[3], selfbleu[4]))
                logging('bleu_test: [{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}]'.format(testbleu[0], testbleu[1], testbleu[2], testbleu[3], testbleu[4]))
                for i in range(5):
                    experiment.log_metric("selfbleu"+str(i+1), selfbleu[i])
                    experiment.log_metric("testbleu"+str(i+1), testbleu[i])
                    #wandb.log({"selfbleu"+str(i+1): selfbleu[i], "testbleu"+str(i+1): testbleu[i]})

            # eval with rev_ppl and for_ppl
            # rev_ppl, for_ppl = train_lm(args.data_path)
            # rev_ppl, for_ppl = 0, 0
            # logging("Epoch {:03d}, Reverse perplexity {}".format(epoch, rev_ppl))
            # logging("Epoch {:03d}, Forward perplexity {}".format(epoch, for_ppl))
            # if best_rev_ppl is None or rev_ppl < best_rev_ppl:
            #     impatience = 0
            #     best_rev_ppl = rev_ppl
            #     logging("New saving model: epoch {:03d}.".format(epoch))
            #     save_model()
            # else:
            #     if not args.no_earlystopping and epoch >= args.min_epochs:
            #         impatience += 1
            #         if impatience > args.patience:
            #             logging("Ending training")
            #             sys.exit()
            if epoch % 15 == 0 or epoch == args.epochs-1:  #This is the last epoch
                logging("New saving model: epoch {:03d}.".format(epoch))
                experiment.log_asset_folder(args.save, step=None, log_file_name=args.save, recursive=False)
                save_model()

    train()
