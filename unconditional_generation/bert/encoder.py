import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from onmt.encoders.encoder import EncoderBase
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

class BertEncoder(nn.Module):
    def __init__(self, vocab, add_noise, num_layers, d_model, heads, d_ff, dropout,attention_dropout, embeddings, max_relative_positions, aehidden):
        super(BertEncoder, self).__init__()
        self.aehidden = aehidden
        self.embeddings = embeddings

        config = BertConfig(num_hidden_layers=num_layers, hidden_size=d_model, num_attention_heads=heads, attention_probs_dropout_prob=attention_dropout, hidden_dropout_prob=dropout, max_position_embeddings=max_relative_positions)
        self.bert = BertModel(config)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if add_noise==True:
            self.squeeze_hidden = nn.Linear(d_model, aehidden*2)
        elif add_noise==False:
            self.squeeze_hidden = nn.Linear(d_model, aehidden)

        self.activation = nn.Tanh()

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, add_noise, soft, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        #self._check_args(src, lengths)
        emb = self.embeddings(src, soft=soft)  #emb [16,64,512] = [max_len, batchsize, d_emb]
        # max_len = lengths.max()  #added by shizhe
        max_len = src.shape[0] ## added by shizhe
        batch_size = src.shape[1]
        # print(max_len)
        out = emb.transpose(0, 1).contiguous()  #out [64,33, 512]
        # if(max_len!=16):
        #     print(max_len)
        mask = ~sequence_mask(lengths, max_len).unsqueeze(1)
        # mask = ~sequence_mask(lengths).unsqueeze(1) #(64,1,33)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.bert._modules:
            out = layer(out, mask)

        if add_noise==True:
            out = self.layer_norm(out)   #out [64, 33, 512]
            memory_bank_ori = out.transpose(0, 1).contiguous()  # [33,64, 512]
            memory_bank_ori = self.squeeze_hidden(memory_bank_ori)  #The original encoder output [33,64,200]
            memory_bank_ori = self.activation(memory_bank_ori)
            #@shizhe add noise
            noise = torch.ones(max_len, batch_size, self.aehidden).normal_(0, 1).cuda()  #[33, 64, 100]
            mean = memory_bank_ori.view(max_len, batch_size, 2, self.aehidden)[:, :, 0]  #[33,64,100]
            var = memory_bank_ori.view(max_len, batch_size, 2, self.aehidden)[:, :, 1]  #[33,64,100]
            memory_bank = mean + var * noise
            # memory_bank = self.activation(self.squeeze_hidden(memory_bank))  #[16,64,300]
            # memory_bank = self.unsqueeze_hidden(memory_bank)  # [16,64,512]   put this line outside the encoder for optimizing reason

            # return emb, out.transpose(0, 1).contiguous(), lengths
        elif add_noise==False:
            out = self.layer_norm(out)
            memory_bank = out.transpose(0, 1).contiguous()
            memory_bank = self.squeeze_hidden(memory_bank)  # [16,64,100]
            memory_bank = self.activation(memory_bank)

        return emb, memory_bank, lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.bert._modules:
            layer.update_dropout(dropout, attention_dropout)
