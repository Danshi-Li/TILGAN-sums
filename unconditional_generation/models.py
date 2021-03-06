import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from lang.onmt.encoders.transformer import TransformerEncoder
# from lang.onmt.decoders.transformer import TransformerDecoder
# from lang.onmt.modules.embeddings import *
from onmt.encoders.transformer import TransformerEncoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.modules.embeddings import *
from utils import to_gpu
from torch.nn.init import xavier_uniform_
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, GPT2Config, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, EncoderDecoderModel
from bert.encoder import BertEncoder, BertEmbeddings, BertPooler
torch.set_printoptions(profile="full")

#from gpt.tokenization_gpt2 import GPT2Tokenizer
from gpt.modeling_gpt2 import GPT2ForLatentConnector

class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=True):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

class MLP_D_local(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=True):
        super(MLP_D_local, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        # x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=True, gan_g_activation=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gan_g_activation = gan_g_activation
        self.tanh = nn.Tanh()

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.gan_g_activation is True:
            x = self.tanh(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq(nn.Module):
    def __init__(self, add_noise, emsize, nhidden, ntokens, nlayers, nheads, nff, aehidden, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=True):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))
        # Transformer Embedding
        self.embedding = Embeddings(
            word_vec_size=emsize,
            position_encoding=True,
            word_padding_idx=0,
            word_vocab_size=ntokens,
        )

        # Transformer Encoder and Decoder
        # nheads = 8
        # nff = 2048
        # aehidden = 200
        atten_dropout = dropout
        max_rela_posi = 0
        copyatten = False
        selfattntype = "scaled-dot"
        aanuseffn = False
        fullcontextalignment=False
        alignmentlayer=0
        alignmentheads=0
        self.encoder = TransformerEncoder(add_noise, nlayers, nhidden, nheads, nff, dropout, atten_dropout, self.embedding, max_rela_posi, aehidden)
        self.unsqueeze_hidden = nn.Linear(aehidden, nhidden)
        self.decoder = TransformerDecoder(nlayers, nhidden, nheads, nff, copyatten, selfattntype, dropout, atten_dropout, self.embedding, max_rela_posi, aanuseffn,fullcontextalignment, alignmentlayer, alignmentheads)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()


    def init_weights(self):
        # init for RNN, but I do not use this function because of transformer. not sure it is ok or not  @shizhe
        initrange = 0.1

        """Initiate parameters in the transformer model."""

        for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)


    def forward(self, indices, lengths, target, add_noise, soft, encode_only=False):
        # batch_size, maxlen = indices.size()
        #
        # hidden = self.encode(indices, lengths, noise)
        #
        # if encode_only:
        #     return hidden
        #
        # if hidden.requires_grad:
        #     hidden.register_hook(self.store_grad_norm)
        #
        # decoded = self.decode(hidden, batch_size, maxlen,
        #                       indices=indices, lengths=lengths)
        #
        # return decoded
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        batchsize = indices.shape[0]  #64
        max_len = indices.shape[1] #16
        # src = self.embedding(indices)
        # src = pack_padded_sequence(input=embeddings,lengths=lengths,batch_first=True)
        src = indices.transpose(0, 1) #[16,64] = [max_len, batchsize]
        # tgt = indices.transpose(0, 1) #[16,64] = [max_len, batchsize]
        tgt = target.view(batchsize, max_len).transpose(0,1)
        if soft==False:
            src = src.unsqueeze(2)
            tgt = tgt.unsqueeze(2)
        # src = src.unsqueeze(2)
        # tgt = tgt.unsqueeze(2)
        # dec_in = tgt[:-1]  # exclude last target from inputs
        if lengths == None:
            lengths_tensor = torch.LongTensor(batchsize)
            lengths_tensor[:] = max_len
        else:
            lengths_tensor = torch.LongTensor(lengths)  #[64]
        #   lengths_tensor = torch.LongTensor(lengths)
        # lengths_tensor[:] = max(lengths_tensor)
        enc_state, memory_bank, lengths = self.encoder(src, add_noise, soft, lengths_tensor) #enc_state=[16,64,512]  memory_back=[16,64,100] lengths=[64]

        if encode_only:
            # return torch.sum(memory_bank, 0)  #[64,512]  doing pooling to produce a single vector
            return memory_bank.transpose(0,1).contiguous().view(batchsize, -1)  #[64, 1600] doing concatenation
        bptt = False
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        memory_bank = self.unsqueeze_hidden(memory_bank)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths_tensor,
                                      with_align=False)
        dec_out = dec_out.transpose(0,1) # dec_out [64,16,512] = [batchsize, max_len, nhidden]
        # reshape to batch_size*maxlen x nhidden before linear over vocab

        decoded = self.linear(dec_out.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batchsize, max_len, self.ntokens)

        # return dec_out, attns
        return decoded


    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        with torch.no_grad():
            self.start_symbols.resize_(batch_size, 1)
            self.start_symbols.fill_(1)

        # embedding = self.embedding_decoder(self.start_symbols)
        # embedding = self.embedding(self.start_symbols)
        # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen)
        # unroll
        all_indices = []
        bptt = False
        if bptt is False:
            self.decoder.init_state(memory_bank, memory_bank, None)
        for step in range(maxlen):
            # print(step)
            # Shape: (1, B, 1)
            # decoder_input = random_sampler.alive_seq[:, -1].view(1, -1, 1)

            # log_probs, attn = self._decode_and_generate(
            #     decoder_input,
            #     memory_bank,
            #     memory_lengths=memory_lengths,
            #     step=step
            # )

            # Decoder forward, takes [tgt_len, batch, nfeats] as input
            # and [src_len, batch, hidden] as memory_bank
            # in case of inference tgt_len = 1, batch = beam times batch_size
            #   i.e. dec_out is in shape of [1, beam_size * batch_size, hidden]
            #        dec_attn is a dict which values are in shape of [1, beam_size * batch_size, src_len]
            # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
            # decoder_input [1,64,1] memory_bank [33,64,512]  memory_length [64]

            dec_out, dec_attn = self.decoder(
                decoder_input, memory_bank, memory_lengths=memory_lengths, step=step
            )
            dec_out = dec_out.transpose(0, 1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            decoded = self.linear(dec_out)   #[64,ntokens]
            # decoded = decoded.view(batch_size, 1, self.ntokens) #[64, 1, ntokens]

            if not sample:
                vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
                indices = indices.unsqueeze(1) #[64,1]
            else:
                probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
                indices = torch.multinomial(probs, 1) #[64,1]

            all_indices.append(indices)

            # embedding = self.embedding(indices)
            # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
            decoder_input = indices.unsqueeze(0)

        # dec_out, attns = self.decoder(hidden, hidden,
        #                               memory_lengths=lengths_tensor,
        #                               with_align=False)
        # for i in range(maxlen):
        #     output, state = self.decoder(inputs, state)
        #     overvocab = self.linear(dec_out.squeeze(1))
        #     dec_out = dec_out.transpose(0, 1)
        #     if not sample:
        #         vals, indices = torch.max(overvocab, 1)
        #     else:
        #         probs = F.softmax(overvocab / temp, dim=-1)
        #         indices = torch.multinomial(probs, 1)
        #     indices = indices.unsqueeze(1)
        #     all_indices.append(indices)
        #
        #     embedding = self.embedding_decoder(indices)
        #     inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)
        return max_indices


    def generate_enh_dec(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        self.start_symbols.resize_(batch_size, 1)
        self.start_symbols.fill_(1)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen+1)
        # unroll
        all_indices = []
        bptt = False
        if bptt is False:
            self.decoder.init_state(memory_bank, memory_bank, None)
        for step in range(maxlen+1):
            dec_out, dec_attn = self.decoder(
                decoder_input, memory_bank, memory_lengths=memory_lengths, step=step
            )
            dec_out = dec_out.transpose(0, 1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            decoded = self.linear(dec_out)   #[64,ntokens]
            # if not sample:
            #     vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
            #     indices = indices.unsqueeze(1) #[64,1]
            # else:
            #     probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
            #     indices = torch.multinomial(probs, 1) #[64,1]
            probs = F.softmax(decoded / temp, dim=-1)  # [64,3455]    [batch_size, num_tokens]
            indices = torch.multinomial(probs, 1)  # [64,1]
            if step == 0:
                decoder_output = probs.unsqueeze(0)  # [1, 64,3455]
            else:
                probs = probs.unsqueeze(0) #[1, 64, 3455]
                decoder_output = torch.cat([decoder_output, probs], 0)  #[33, 64, 3455] = [max_len, batch_size, num_tokens]
            all_indices.append(indices)
            decoder_input = indices.unsqueeze(0)

        # decoder_output = decoder_output.transpose(0,1).reshape(batch_size, -1) #[64, 33 * 3455] = [batch_size, max_len * num_tokens]
        decoder_output = decoder_output.transpose(0, 1)  # [64, 32, 3455] = [batch_size, max_len, num_tokens]
        max_indices = torch.cat(all_indices, 1)

        return decoder_output, max_indices # decoder_output: [64, 33, 3455]   max_indices: [64, 32]



class AE_BERT_enc(nn.Module):
    def __init__(self, vocab, add_noise, emsize, nhidden, ntokens, nlayers, nheads, nff, aehidden, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=True):
        super(AE_BERT_enc, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.config = BertConfig(num_hidden_layers=nlayers, hidden_size=aehidden, num_attention_heads=nheads, attention_probs_dropout_prob=dropout, hidden_dropout_prob=dropout, max_position_embeddings=emsize)
        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))
        
        # Transformer embedding
        self.enc_embedding = BertEmbeddings(self.config)
        
        self.dec_embedding = Embeddings(
            word_vec_size=emsize,
            position_encoding=True,
            word_padding_idx=0,
            word_vocab_size=ntokens,
        )
        
        # Transformer Encoder and Decoder
        # nheads = 8
        # nff = 2048
        # aehidden = 200
        atten_dropout = dropout
        max_rela_posi = 0
        copyatten = False
        selfattntype = "scaled-dot"
        aanuseffn = False
        fullcontextalignment=False
        alignmentlayer=0
        alignmentheads=0

        self.encoder = BertEncoder(self.config)
        self.unsqueeze_hidden = nn.Linear(aehidden, nhidden)
        self.decoder = TransformerDecoder(nlayers, nhidden, nheads, nff, copyatten, selfattntype, dropout, atten_dropout, self.dec_embedding, max_rela_posi, aanuseffn,fullcontextalignment, alignmentlayer, alignmentheads)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()


    def init_weights(self):
        # init for RNN, but I do not use this function because of transformer. not sure it is ok or not  @shizhe
        initrange = 0.1

        """Initiate parameters in the transformer model."""

        for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)


    def forward(self, indices, lengths, target, add_noise, soft, encode_only=False):
        # batch_size, maxlen = indices.size()
        #
        # hidden = self.encode(indices, lengths, noise)
        #
        # if encode_only:
        #     return hidden
        #
        # if hidden.requires_grad:
        #     hidden.register_hook(self.store_grad_norm)
        #
        # decoded = self.decode(hidden, batch_size, maxlen,
        #                       indices=indices, lengths=lengths)
        #
        # return decoded
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        batchsize = indices.shape[0]  #64
        max_len = indices.shape[1] #16
        #src = self.embedding(indices)
        # src = pack_padded_sequence(input=embeddings,lengths=lengths,batch_first=True)
        src = indices.transpose(0, 1) #[16,64] = [max_len, batchsize]
        # tgt = indices.transpose(0, 1) #[16,64] = [max_len, batchsize]
        tgt = target.view(batchsize, max_len).transpose(0,1)
        
        if soft==False:
            tgt=tgt.unsqueeze(2)
        
        # dec_in = tgt[:-1]  # exclude last target from inputs
        if lengths == None:
            lengths_tensor = torch.LongTensor(batchsize)
            lengths_tensor[:] = max_len
        else:
            lengths_tensor = torch.LongTensor(lengths)  #[64]
        #   lengths_tensor = torch.LongTensor(lengths)
        # lengths_tensor[:] = max(lengths_tensor)
        #enc_state, memory_bank, lengths = self.encoder(src, add_noise, soft, lengths_tensor) #enc_state=[16,64,512]  memory_back=[16,64,100] lengths=[64]
        
        src = self.enc_embedding(src, soft=soft)
        
        memory_bank = self.encoder(src)[0]
        #print("Successfully attained latent output from encoder")
        if encode_only:
            # return torch.sum(memory_bank, 0)  #[64,512]  doing pooling to produce a single vector
            return memory_bank.transpose(0,1).contiguous().view(batchsize, -1)  #[64, 1600] doing concatenation
        bptt = False
        if bptt is False:
            self.decoder.init_state(src, memory_bank, None)
        memory_bank = self.unsqueeze_hidden(memory_bank)
        
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths_tensor,
                                      with_align=False)
        dec_out = dec_out.transpose(0,1) # dec_out [64,16,512] = [batchsize, max_len, nhidden]
        # reshape to batch_size*maxlen x nhidden before linear over vocab

        decoded = self.linear(dec_out.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batchsize, max_len, self.ntokens)

        # return dec_out, attns
        return decoded


    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        with torch.no_grad():
            self.start_symbols.resize_(batch_size, 1)
            self.start_symbols.fill_(1)

        # embedding = self.embedding_decoder(self.start_symbols)
        # embedding = self.embedding(self.start_symbols)
        # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen)
        # unroll
        all_indices = []
        bptt = False
        if bptt is False:
            self.decoder.init_state(memory_bank, memory_bank, None)
        for step in range(maxlen):
            # print(step)
            # Shape: (1, B, 1)
            # decoder_input = random_sampler.alive_seq[:, -1].view(1, -1, 1)

            # log_probs, attn = self._decode_and_generate(
            #     decoder_input,
            #     memory_bank,
            #     memory_lengths=memory_lengths,
            #     step=step
            # )

            # Decoder forward, takes [tgt_len, batch, nfeats] as input
            # and [src_len, batch, hidden] as memory_bank
            # in case of inference tgt_len = 1, batch = beam times batch_size
            #   i.e. dec_out is in shape of [1, beam_size * batch_size, hidden]
            #        dec_attn is a dict which values are in shape of [1, beam_size * batch_size, src_len]
            # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
            # decoder_input [1,64,1] memory_bank [33,64,512]  memory_length [64]

            dec_out, dec_attn = self.decoder(
                decoder_input, memory_bank, memory_lengths=memory_lengths, step=step
            )
            dec_out = dec_out.transpose(0, 1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            decoded = self.linear(dec_out)   #[64,ntokens]
            # decoded = decoded.view(batch_size, 1, self.ntokens) #[64, 1, ntokens]

            if not sample:
                vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
                indices = indices.unsqueeze(1) #[64,1]
            else:
                probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
                indices = torch.multinomial(probs, 1) #[64,1]

            all_indices.append(indices)

            # embedding = self.embedding(indices)
            # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
            decoder_input = indices.unsqueeze(0)

        # dec_out, attns = self.decoder(hidden, hidden,
        #                               memory_lengths=lengths_tensor,
        #                               with_align=False)
        # for i in range(maxlen):
        #     output, state = self.decoder(inputs, state)
        #     overvocab = self.linear(dec_out.squeeze(1))
        #     dec_out = dec_out.transpose(0, 1)
        #     if not sample:
        #         vals, indices = torch.max(overvocab, 1)
        #     else:
        #         probs = F.softmax(overvocab / temp, dim=-1)
        #         indices = torch.multinomial(probs, 1)
        #     indices = indices.unsqueeze(1)
        #     all_indices.append(indices)
        #
        #     embedding = self.embedding_decoder(indices)
        #     inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)
        return max_indices


    def generate_enh_dec(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        self.start_symbols.resize_(batch_size, 1)
        self.start_symbols.fill_(1)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen+1)
        # unroll
        all_indices = []
        bptt = False
        if bptt is False:
            self.decoder.init_state(memory_bank, memory_bank, None)
        for step in range(maxlen+1):
            dec_out, dec_attn = self.decoder(
                decoder_input, memory_bank, memory_lengths=memory_lengths, step=step
            )
            dec_out = dec_out.transpose(0, 1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            decoded = self.linear(dec_out)   #[64,ntokens]
            # if not sample:
            #     vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
            #     indices = indices.unsqueeze(1) #[64,1]
            # else:
            #     probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
            #     indices = torch.multinomial(probs, 1) #[64,1]
            probs = F.softmax(decoded / temp, dim=-1)  # [64,3455]    [batch_size, num_tokens]
            indices = torch.multinomial(probs, 1)  # [64,1]
            if step == 0:
                decoder_output = probs.unsqueeze(0)  # [1, 64,3455]
            else:
                probs = probs.unsqueeze(0) #[1, 64, 3455]
                decoder_output = torch.cat([decoder_output, probs], 0)  #[33, 64, 3455] = [max_len, batch_size, num_tokens]
            all_indices.append(indices)
            decoder_input = indices.unsqueeze(0)

        # decoder_output = decoder_output.transpose(0,1).reshape(batch_size, -1) #[64, 33 * 3455] = [batch_size, max_len * num_tokens]
        decoder_output = decoder_output.transpose(0, 1)  # [64, 32, 3455] = [batch_size, max_len, num_tokens]
        max_indices = torch.cat(all_indices, 1)

        return decoder_output, max_indices # decoder_output: [64, 33, 3455]   max_indices: [64, 32]


class AE_GPT_dec(nn.Module):
    def __init__(self, add_noise, emsize, nhidden, ntokens, gpt_tokens, nlayers, nheads, nff, aehidden, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=True):
        super(AE_GPT_dec, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.gpt_tokens = gpt_tokens

        self.config = GPT2Config.from_pretrained("gpt2")
        self.config.n_positions = 2048
        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))
        # Transformer Embedding
        self.enc_embedding = Embeddings(
            word_vec_size=emsize,
            position_encoding=True,
            word_padding_idx=0,
            word_vocab_size=ntokens,
        )
        self.dec_embedding = None

        # Transformer Encoder and Decoder
        # nheads = 8
        # nff = 2048
        # aehidden = 200
        atten_dropout = dropout
        max_rela_posi = 0
        copyatten = False
        selfattntype = "scaled-dot"
        aanuseffn = False
        fullcontextalignment=False
        alignmentlayer=0
        alignmentheads=0

        self.encoder = TransformerEncoder(add_noise, nlayers, nhidden, nheads, nff, dropout, atten_dropout, self.enc_embedding, max_rela_posi, aehidden)
        self.unsqueeze_hidden = nn.Linear(aehidden, 768)
        self.decoder = GPT2ForLatentConnector.from_pretrained("gpt2")

        # Initialize Linear Transformation
        self.linear = nn.Linear(768, gpt_tokens)

        self.init_weights()


    def init_weights(self):
        # init for RNN, but I do not use this function because of transformer. not sure it is ok or not  @shizhe
        initrange = 0.1

        """Initiate parameters in the transformer model."""

        for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)


    def forward(self, indices, lengths, target, add_noise, soft, encode_only=False):
        # batch_size, maxlen = indices.size()
        #
        # hidden = self.encode(indices, lengths, noise)
        #
        # if encode_only:
        #     return hidden
        #
        # if hidden.requires_grad:
        #     hidden.register_hook(self.store_grad_norm)
        #
        # decoded = self.decode(hidden, batch_size, maxlen,
        #                       indices=indices, lengths=lengths)
        #
        # return decoded
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        batchsize = indices.shape[0]  #64
        max_len = indices.shape[1] #16
        # src = self.embedding(indices)
        # src = pack_padded_sequence(input=embeddings,lengths=lengths,batch_first=True)
        src = indices.transpose(0, 1) #[16,64] = [max_len, batchsize]
        # tgt = indices.transpose(0, 1) #[16,64] = [max_len, batchsize]
        tgt = target.view(batchsize, max_len).transpose(0,1)
        if soft==False:
            src = src.unsqueeze(2)
            #tgt = tgt.unsqueeze(2)
        # src = src.unsqueeze(2)
        # tgt = tgt.unsqueeze(2)
        # dec_in = tgt[:-1]  # exclude last target from inputs
        if lengths == None:
            lengths_tensor = torch.LongTensor(batchsize)
            lengths_tensor[:] = max_len
        else:
            lengths_tensor = torch.LongTensor(lengths)  #[64]
        #   lengths_tensor = torch.LongTensor(lengths)
        # lengths_tensor[:] = max(lengths_tensor)
        enc_state, memory_bank, lengths = self.encoder(src, add_noise, soft, lengths_tensor) #enc_state=[16,64,512]  memory_back=[16,64,100] lengths=[64]
        memory_bank = self.unsqueeze_hidden(memory_bank)
        if encode_only:
            # return torch.sum(memory_bank, 0)  #[64,512]  doing pooling to produce a single vector
            return memory_bank.transpose(0,1).contiguous().view(batchsize, -1)  #[64, 1600] doing concatenation
        '''
        bptt = False
        if bptt is False:
            self.init_state(src, memory_bank, enc_state)
        memory_bank = self.unsqueeze_hidden(memory_bank)
        '''
        #memory_bank = self.dec_embedding(memory_bank)
        memory_bank = memory_bank.contiguous()
        '''
        The method applied by Optimus, see paper fig.3. Different from our design
        Different decoder dims applied
        dec_out = self.decoder(tgt, past=memory_bank)
        '''
        #memory_bank dim: [max_len+1, batch_size, nhidden]
        #tgt dim: [max_len+1, batch_size, 1]
        #memory_bank = memory_bank.view(-1, memory_bank.shape[2])
        #print("tgt size:",tgt.shape)
        #dec_out = self.decoder(input_ids=tgt, past=memory_bank, labels=tgt)
        dec_out = self.decoder(input_ids=tgt,past=memory_bank)
        #dec_out = dec_out.transpose(0,1) # dec_out [64,16,512] = [batchsize, max_len, nhidden]
        # reshape to batch_size*maxlen x nhidden before linear over vocab

        #decoded = self.linear(dec_out.contiguous().view(-1, self.nhidden))
        decoded = dec_out.transpose(0,1)
        # return dec_out, attns
        return decoded


    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        with torch.no_grad():
            self.start_symbols.resize_(batch_size, 1)
            self.start_symbols.fill_(1)

        # embedding = self.embedding_decoder(self.start_symbols)
        # embedding = self.embedding(self.start_symbols)
        # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        #memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen)
        # unroll
        all_indices = []

        for step in range(maxlen):

            dec_out = self.decoder(
                input_ids=decoder_input, past=memory_bank
            )
            dec_out = dec_out.transpose(0, 1).squeeze(1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            #print("dec out dim:",dec_out.shape)
            #print("self linear shape:",self.linear.weight.shape)
            decoded = dec_out  #[64,ntokens]
            # decoded = decoded.view(batch_size, 1, self.ntokens) #[64, 1, ntokens]

            if not sample:
                vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
                indices = indices.unsqueeze(1) #[64,1]
            else:
                probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
                indices = torch.multinomial(probs, 1) #[64,1]

            all_indices.append(indices)

            # embedding = self.embedding(indices)
            # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
            decoder_input = indices.unsqueeze(0)

        max_indices = torch.cat(all_indices, 1)
        return max_indices


    def generate_enh_dec(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        self.start_symbols.resize_(batch_size, 1)
        self.start_symbols.fill_(1)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen+1)
        # unroll
        all_indices = []
        '''
        bptt = False
        if bptt is False:
            self.decoder.init_state(memory_bank, memory_bank, None)
        '''
        for step in range(maxlen+1):
            dec_out, dec_attn = self.decoder(
                input_ids=decoder_input, past=memory_bank)
            dec_out = dec_out.transpose(0, 1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            decoded = self.linear(dec_out)   #[64,ntokens]
            # if not sample:
            #     vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
            #     indices = indices.unsqueeze(1) #[64,1]
            # else:
            #     probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
            #     indices = torch.multinomial(probs, 1) #[64,1]
            probs = F.softmax(decoded / temp, dim=-1)  # [64,3455]    [batch_size, num_tokens]
            indices = torch.multinomial(probs, 1)  # [64,1]
            if step == 0:
                decoder_output = probs.unsqueeze(0)  # [1, 64,3455]
            else:
                probs = probs.unsqueeze(0) #[1, 64, 3455]
                decoder_output = torch.cat([decoder_output, probs], 0)  #[33, 64, 3455] = [max_len, batch_size, num_tokens]
            all_indices.append(indices)
            decoder_input = indices.unsqueeze(0)

        # decoder_output = decoder_output.transpose(0,1).reshape(batch_size, -1) #[64, 33 * 3455] = [batch_size, max_len * num_tokens]
        decoder_output = decoder_output.transpose(0, 1)  # [64, 32, 3455] = [batch_size, max_len, num_tokens]
        max_indices = torch.cat(all_indices, 1)

        return decoder_output, max_indices # decoder_output: [64, 33, 3455]   max_indices: [64, 32]


class AE_BG(nn.Module):
    def __init__(self, add_noise, emsize, nhidden, ntokens, gpt_tokens, nlayers, nheads, nff, aehidden, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=True, decoder_impl='cai', unified_embedding=False):
        super(AE_BG, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = 50257    #vocab_size for GPT model
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.gpt_tokens = gpt_tokens
        self.Bertconfig = BertConfig(num_hidden_layers=nlayers, hidden_size=aehidden, num_attention_heads=nheads, attention_probs_dropout_prob=dropout, hidden_dropout_prob=dropout, max_position_embeddings=emsize)

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))
        # Transformer Embedding
        self.enc_embedding = BertEmbeddings(self.Bertconfig)
        self.dec_embedding = None

        if unified_embedding:
            self.tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
        
        
        
        if decoder_impl=='tutorial':
            # encoder-decoder implementation with the transformers.EncoderDecoderModel() class
            self.autoencoder = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased","gpt2")
            self.encoder = self.autoencoder.get_encoder()
            self.decoder = self.autoencoder.get_decoder()
        else:
            #encoder-decoder implementation with separate classes
            self.encoder = BertModel.from_pretrained("bert-base-cased")
            self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        


        #self.init_weights()


    def init_weights(self):
        # init for RNN, but I do not use this function because of transformer. not sure it is ok or not  @shizhe
        initrange = 0.1

        """Initiate parameters in the transformer model."""
        '''
        for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        '''
        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)


    def forward(self, indices, lengths, target, add_noise, soft, encode_only=False, decoder_impl='cai', unified_embedding=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        batch_size = indices.shape[0]  #64
        max_len = indices.shape[1] #16
        src = indices.contiguous() #[16,64] = [max_len, batchsize]
        tgt = target.contiguous()
        #print("src:",src)
        #print("tgt:",tgt)
        
        if unified_embedding:
            enc_input_emb = self.tokenizer(src)
            memory_bank = self.encoder(inputs_embeds=enc_input_emb).last_hidden_state.contiguous()
        else:
            memory_bank = self.encoder(input_ids=src).last_hidden_state.contiguous()
        #print("Successfully attained latent output from encoder")
        if encode_only:
            return memory_bank.transpose(0,1).contiguous().view(batch_size, -1)  #[64, 1600] doing concatenation
        #memory_bank = self.unsqueeze_hidden(memory_bank)

        '''
        The method applied by Optimus, see paper fig.3. Different from our design
        Different decoder dims applied
        dec_out = self.decoder(tgt, past=memory_bank).logits
        '''
        #memory_bank dim: [max_len+1, batch_size, nhidden]
        #tgt dim: [max_len+1, batch_size, 1]
        '''
        
        '''
        if decoder_impl == 'cai':
            #The 'cai' implementation concats representation of [CLS] in BERT last layer with shifted target text as input to GPT
            tgt_embed = self.decoder.transformer.wte(tgt[:,1:])
            #using the representation of [CLS] as sentence embedding
            memory_bank = memory_bank[:,0,:].unsqueeze(1)
            decoder_input = torch.cat((memory_bank,tgt_embed),1).to(memory_bank.device)
            #print("decoder_input shape:",decoder_input.shape)
            dec_out = self.decoder(inputs_embeds=decoder_input).logits
        elif decoder_impl== 'cai-stepwise':
            #As described in the paper of Cai et al, use last layer [CLS] representation as zeroth token into GPT and generate text stepwise
            #delete the ending [SEP] for the target sequence input to decoder
            decoder_input = memory_bank[:,0,:].unsqueeze(1)
            dec_out = []
            decoded = self.decoder(inputs_embeds=decoder_input).logits
            decoded = decoded.view(batch_size, 1, self.gpt_tokens)
            dec_out.append(decoded)
            vals, indices = torch.max(decoded, -1)  #vals: [64], indices: [64]
            decoder_input = indices.view(batch_size,1).contiguous()
            for step in range(max_len-1):
                decoded = self.decoder(input_ids=decoder_input).logits
                decoded = decoded.view(batch_size, 1, self.gpt_tokens)

                dec_out.append(decoded)
                vals, indices = torch.max(decoded, -1) #vals: [64], indices: [64]
                decoder_input = indices.view(batch_size,1).contiguous()
            dec_out = torch.cat(dec_out,1)
        elif decoder_impl == 'bankonly':
            # The second implementation uses the whole last layer BERT state as input embedding to GPT
            dec_out = self.decoder(inputs_embeds=memory_bank).logits
        elif decoder_impl == 'tutorial':
            #The third implementation is adapted from the tutorial @shizhe shared and utilizes BERT last layer hidden state to interact with GPT in cross-attention layers
            dec_out = self.decoder(input_ids=tgt,encoder_hidden_states=memory_bank).logits.contiguous()


        #decoded = self.linear(dec_out.contiguous())
        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = dec_out.view(batch_size, max_len, self.gpt_tokens).contiguous()
        return decoded


    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        with torch.no_grad():
            self.start_symbols.resize_(batch_size, 1)
            self.start_symbols.fill_(1)

        # embedding = self.embedding_decoder(self.start_symbols)
        # embedding = self.embedding(self.start_symbols)
        # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous()  #[64,3300] -> [64,33,100] -> [33,64,100]
        #memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen)
        # unroll
        all_indices = []

        decoded = self.decoder(
                    inputs_embeds=memory_bank[:,0,:].unsqueeze(1)
                ).logits
        decoded = decoded.view(batch_size, 1, self.ntokens)
        if not sample:
            vals, indices = torch.max(decoded, -1)  #vals: [64], indices: [64]
            indices = indices.unsqueeze(1) #[64,1]
        else:
            probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
            indices = torch.multinomial(probs, -1) #[64,1]
        indices = indices.squeeze(-1)
        all_indices.append(indices)
        decoder_input = indices


        for step in range(maxlen):
            #print("generate input step {}:".format(step),decoder_input)
            decoded = self.decoder(input_ids=decoder_input).logits.contiguous()
            #decoded = self.linear(decoded).contiguous()
            #print("dec out dim:",dec_out.shape)
            #print("self linear shape:",self.linear.weight.shape)
            decoded = decoded.view(batch_size, 1, self.gpt_tokens) #[64, 1, ntokens]

            if not sample:
                vals, indices = torch.max(decoded, -1)  #vals: [64], indices: [64]
                indices = indices.unsqueeze(1) #[64,1]
            else:
                probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
                indices = torch.multinomial(probs, -1) #[64,1]

            indices = indices.squeeze(-1)
            all_indices.append(indices)

            # embedding = self.embedding(indices)
            # inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)
            decoder_input = indices
            

        max_indices = torch.cat(all_indices, 1)
        return max_indices


    def generate_enh_dec(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        #hidden: [64, 3300]
        batch_size = hidden.size(0)  #64

        # <sos>
        self.start_symbols.resize_(batch_size, 1)
        self.start_symbols.fill_(1)
        decoder_input = self.start_symbols.transpose(0,1).unsqueeze(2) #[1,64,1]
        memory_bank = hidden.view(batch_size, maxlen+1, -1).contiguous().transpose(0,1)  #[64,3300] -> [64,33,100] -> [33,64,100]
        memory_bank = self.unsqueeze_hidden(memory_bank)  # [33,64,100] -> [33,64,512]
        # memory_bank = hidden.expand(maxlen, hidden.shape[0], hidden.shape[1])  #[15, 64,512]
        memory_lengths = torch.ones(batch_size).fill_(maxlen+1)
        # unroll
        all_indices = []
        '''
        bptt = False
        if bptt is False:
            self.decoder.init_state(memory_bank, memory_bank, None)
        '''
        for step in range(maxlen+1):
            dec_out, dec_attn = self.decoder(
                input_ids=decoder_input, past=memory_bank)
            dec_out = dec_out.transpose(0, 1).squeeze(1)    # dec_out [64,1,512] -> [64,512]
            decoded = self.linear(dec_out)   #[64,ntokens]
            # if not sample:
            #     vals, indices = torch.max(decoded, 1)  #vals: [64], indices: [64]
            #     indices = indices.unsqueeze(1) #[64,1]
            # else:
            #     probs = F.softmax(decoded / temp, dim=-1)  #[64,978]
            #     indices = torch.multinomial(probs, 1) #[64,1]
            probs = F.softmax(decoded / temp, dim=-1)  # [64,3455]    [batch_size, num_tokens]
            indices = torch.multinomial(probs, 1)  # [64,1]
            if step == 0:
                decoder_output = probs.unsqueeze(0)  # [1, 64,3455]
            else:
                probs = probs.unsqueeze(0) #[1, 64, 3455]
                decoder_output = torch.cat([decoder_output, probs], 0)  #[33, 64, 3455] = [max_len, batch_size, num_tokens]
            all_indices.append(indices)
            decoder_input = indices.unsqueeze(0)

        # decoder_output = decoder_output.transpose(0,1).reshape(batch_size, -1) #[64, 33 * 3455] = [batch_size, max_len * num_tokens]
        decoder_output = decoder_output.transpose(0, 1)  # [64, 32, 3455] = [batch_size, max_len, num_tokens]
        max_indices = torch.cat(all_indices, 1)

        return decoder_output, max_indices # decoder_output: [64, 33, 3455]   max_indices: [64, 32]
