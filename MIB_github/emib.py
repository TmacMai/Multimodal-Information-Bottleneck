import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from modules.transformer import TransformerEncoder

import torch
import torch.utils.checkpoint
from torch import nn
#from torch.nn import CrossEntropyLoss, MSELoss

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

from transformers.modeling_bert import BertPreTrainedModel
from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig

import torch.optim as optim
from itertools import chain

from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, DEVICE

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_new": gelu_new,
    "mish": mish,
}


BertLayerNorm = torch.nn.LayerNorm


class MIB_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, d_l):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
  
        self.d_l = d_l
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        # visual,
        # acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

       
        fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
    #    print(sequence_output.shape)
  

        outputs = sequence_output.transpose(1, 2)
        outputs = self.proj_l(outputs)
       # outputs1 = torch.mean(outputs, dim=2)
        pooled_output = outputs[:, :, -1]

    #   #  print(pooled_output.shape)
    #     outputs = (sequence_output, pooled_output,) + encoder_outputs[
    #         1:
    #     ]  # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return pooled_output


beta   = 1e-3
from torch.optim import AdamW, Adam
from torch.nn import Sequential
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from torch.autograd import Variable


class MIB(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.d_l = 50

        self.bert = MIB_BertModel(config, multimodal_config, self.d_l)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
     #   self.classifier = nn.Linear(self.d_l, config.num_labels)


        #self.d_a = 50
        self.attn_dropout = 0.5
        self.proj_a = nn.Conv1d(74, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_v = nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_l = nn.Conv1d(768, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.transa = self.get_network(self_type='l', layers=3)
        self.transv = self.get_network(self_type='l', layers=3)
       # self.BN = nn.BatchNorm1d(self.d_l)

        self.llr = 1e-5   
        self.lr = 2e-5   #20e-6
        
        
        self.fusion = fusion(self.d_l)

        

        self.optimizer_l = Adam(self.bert.parameters(), lr=self.llr)
        self.optimizer_all = getattr(optim, 'Adam')(chain(self.transa.parameters(), self.transv.parameters(), self.fusion.parameters(), self.proj_a.parameters(), self.proj_v.parameters()), lr=self.lr)


       # self.optimizer_a = getattr(optim, 'Adam')(chain(self.proj_a.parameters(), self.transa.parameters()), lr=self.lr)
      #  self.optimizer_v = getattr(optim, 'Adam')(chain(self.proj_v.parameters(), self.transv.parameters()), lr=self.lr)



        self.mean = nn.AdaptiveAvgPool1d(1)

      

        self.init_weights()






    def get_network(self, self_type='l', layers=5):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout

        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads= 5, #self.num_heads,
                                  layers=layers,#max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=0.3,#self.relu_dropout,
                                  res_dropout= 0.3,#self.res_dropout,
                                  embed_dropout=0.2,#self.embed_dropout,
                                  attn_mask= False)#self.attn_mask)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        output_l = outputs


        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        output_a = outputa[-1]  # 48 50
        output_v = outputv[-1]


        outputf, loss_u = self.fusion(output_l, output_a, output_v, label_ids)

      #  y_m = self.classifier(outputf)

        loss_fct = L1Loss()

        loss_m = loss_fct(outputf.view(-1,), label_ids.view(-1,))

        loss = loss_u + loss_m

        self.optimizer_l.zero_grad()
        self.optimizer_all.zero_grad()
        loss.backward(retain_graph = True)
        self.optimizer_all.step()
        self.optimizer_l.step()

        return outputf



    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        output_l = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)



        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        output_a = outputa[-1]  # 48 50
        output_v = outputv[-1]

        outputf = self.fusion.test(output_l, output_a, output_v)


     #   y_m = self.classifier(outputf)

        return outputf







class fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.d_l = dim
        
        # build encoder
        self.encoder_l = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  


        self.encoder_a = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  

        self.encoder_v = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  
        self.encoder = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  

        self.fc_mu_l  = nn.Linear(1024, self.d_l) 
        self.fc_std_l = nn.Linear(1024, self.d_l)

        self.fc_mu_a  = nn.Linear(1024, self.d_l) 
        self.fc_std_a = nn.Linear(1024, self.d_l)

        self.fc_mu_v  = nn.Linear(1024, self.d_l) 
        self.fc_std_v = nn.Linear(1024, self.d_l)

        self.fc_mu  = nn.Linear(1024, self.d_l) 
        self.fc_std = nn.Linear(1024, self.d_l)
        
        # build decoder
        self.decoder_l = nn.Linear(self.d_l, 1)
        self.decoder_a = nn.Linear(self.d_l, 1)
        self.decoder_v = nn.Linear(self.d_l, 1)
        self.decoder = nn.Linear(self.d_l, 1)

      #  self.fusion1 = graph_fusion(self.d_l, self.d_l)
        self.fusion1 = concat(self.d_l, self.d_l)
       # self.fusion1 = tensor(self.d_l, self.d_l)
       # self.fusion1 = addition(self.d_l, self.d_l)
       # self.fusion1 = multiplication(self.d_l, self.d_l)
      #  self.fusion1 = low_rank(self.d_l, self.d_l)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)


    def encode_l(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_l(x)
        return self.fc_mu_l(x), F.softplus(self.fc_std_l(x)-5, beta=1)

    def encode_a(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_a(x)
        return self.fc_mu_a(x), F.softplus(self.fc_std_a(x)-5, beta=1)

    def encode_v(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_v(x)
        return self.fc_mu_v(x), F.softplus(self.fc_std_v(x)-5, beta=1)
    
    def decode_l(self, z):

        return self.decoder_l(z)

    def decode_a(self, z):

        return self.decoder_a(z)

    def decode(self, z):

        return self.decoder(z)

    def decode_v(self, z):

        return self.decoder_v(z)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def loss_function(self, y_pred, y, mu, std):
   
        loss_fct = L1Loss()

        CE = loss_fct(y_pred.view(-1,), y.view(-1,))
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (beta*KL + CE) 


    def loss_function2(self, y_pred, y):
   
        loss_fct = L1Loss()

        CE = loss_fct(y_pred.view(-1,), y.view(-1,))
      #  KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return CE


    def forward(
        self,
        x_l,
        x_a,
        x_v,
        label_ids
    ):



        outputf = self.fusion1(x_l, x_a, x_v)
       # output =  self.decode(outputf)
       
        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)
       
 
        loss = self.loss_function(output, label_ids, mu, std)

        return output, loss


    def test(
        self,
        x_l,
        x_a,
        x_v
    ):

        '''
        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        output_l =  self.decode_l(z_l)
 


        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        output_a =  self.decode_a(z_a)


        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)
        output_v =  self.decode_v(z_v)
        '''
        outputf = self.fusion1(x_l, x_a, x_v)

        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)
    
      #  output =  self.decode(outputf)

        return output




class graph_fusion(nn.Module):

    def __init__(self, in_size, output_dim, hidden = 50, dropout=0.5):

        super(graph_fusion, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size*3)
        self.drop = nn.Dropout(p=dropout)
      #  self.graph = nn.Linear(in_size*2, in_size)

        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )


        self.graph_fusion2 = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size*3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
    #    self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
      #  self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

      #  self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
     #   xavier_normal(self.u1)

    def forward(self, l1, a1, v1):
       # a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
        ###################### unimodal layer  ##########################
        sa = torch.tanh(self.attention(a1))
        sv = torch.tanh(self.attention(v1))
        sl = torch.tanh(self.attention(l1))

        total_weights = torch.cat([sa, sv, sl],1)
      #  total_weights = torch.cat([total_weights,sl],1)

        unimodal_a = (sa.expand(a1.size(0),self.in_size))
        unimodal_v = (sv.expand(a1.size(0),self.in_size))
        unimodal_l = (sl.expand(a1.size(0),self.in_size))
        sa = sa.squeeze(1)
        sl = sl.squeeze(1)
        sv = sv.squeeze(1)
        unimodal = (unimodal_a * a1 + unimodal_v * v1 + unimodal_l * l1)/3

        ##################### bimodal layer ############################
        a = F.softmax(a1, 1)
        v = F.softmax(v1, 1)
        l = F.softmax(l1, 1)
        sav = (1/(torch.matmul(a.unsqueeze(1), v.unsqueeze(2)).squeeze() +0.5) *(sa+sv))
        sal = (1/(torch.matmul(a.unsqueeze(1), l.unsqueeze(2)).squeeze() +0.5) *(sa+sl))
        svl = (1/(torch.matmul(v.unsqueeze(1), l.unsqueeze(2)).squeeze() +0.5) *(sl+sv))

        normalize = torch.cat([sav.unsqueeze(1), sal.unsqueeze(1), svl.unsqueeze(1)],1)
        normalize = F.softmax(normalize,1)
        total_weights = torch.cat([total_weights,normalize],1)

        a_v = torch.tanh((normalize[:,0].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([a1,v1],1)))
        a_l = torch.tanh((normalize[:,1].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([a1,l1],1)))
        v_l = torch.tanh((normalize[:,2].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([v1,l1],1)))
        bimodal = (a_v + a_l + v_l)
    
        ###################### trimodal layer ####################################
        a_v2 = F.softmax(self.graph_fusion(torch.cat([a1,v1],1)), 1)
        a_l2 = F.softmax(self.graph_fusion(torch.cat([a1,l1],1)), 1)
        v_l2 = F.softmax(self.graph_fusion(torch.cat([v1,l1],1)), 1)
        savvl = (1/(torch.matmul(a_v2.unsqueeze(1), v_l2.unsqueeze(2)).squeeze() +0.5) *(sav+svl))
        saavl = (1/(torch.matmul(a_v2.unsqueeze(1), a_l2.unsqueeze(2)).squeeze() +0.5) *(sav+sal))
        savll = (1/(torch.matmul(a_l2.unsqueeze(1), v_l2.unsqueeze(2)).squeeze() +0.5) *(sal+svl))
        savl = (1/(torch.matmul(a_v2.unsqueeze(1), l.unsqueeze(2)).squeeze() +0.5) *(sav+sl))
        salv = (1/(torch.matmul(a_l2.unsqueeze(1), v.unsqueeze(2)).squeeze() +0.5) *(sal+sv))
        svla = (1/(torch.matmul(v_l2.unsqueeze(1), a.unsqueeze(2)).squeeze() +0.5) *(sa+svl))

        normalize2 = torch.cat([savvl.unsqueeze(1), saavl.unsqueeze(1), savll.unsqueeze(1), savl.unsqueeze(1), salv.unsqueeze(1), svla.unsqueeze(1)],1)
        normalize2 = F.softmax(normalize2,1)
        total_weights = torch.cat([total_weights,normalize2],1)
        avvl = torch.tanh((normalize2[:,0].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_v,v_l],1)))
        aavl = torch.tanh((normalize2[:,1].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_v,a_l],1)))
        avll = torch.tanh((normalize2[:,2].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([v_l,a_l],1)))
        avl = torch.tanh((normalize2[:,3].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_v,l1],1)))
        alv = torch.tanh((normalize2[:,4].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_l,v1],1)))
        vla = torch.tanh((normalize2[:,5].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([v_l,a1],1)))
        trimodal = (avvl + aavl + avll + avl + alv + vla)
        fusion = torch.cat([unimodal,bimodal,trimodal],1)

        y_1 = torch.tanh(self.linear_1(fusion))
        y_1 = torch.tanh(self.linear_2(y_1))
        y_2 = torch.tanh(self.linear_3(y_1))

        return y_2


class tensor(nn.Module):

    def __init__(self, in_size, output_dim, hidden = 50, dropout=0.5):

        super(tensor, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, 1)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
        xavier_normal(self.u1)

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, l1, a1, v1):
        DTYPE = torch.cuda.FloatTensor
       # a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
           
        _audio_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), a1), dim=1)
        _video_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), v1), dim=1)
        _text_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), l1), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (a1.size(1) + 1) * (a1.size(1) + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(a1.size(0), -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = (self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = torch.tanh(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.relu(self.post_fusion_layer_3(post_fusion_y_2))
      #  output = post_fusion_y_3 * self.output_range + self.output_shift
        y_2 = post_fusion_y_3


        return y_2



class low_rank(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(low_rank, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, 1)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.rank = 5
        self.output_dim = output_dim
        self.audio_factor = Parameter(torch.Tensor(self.rank, in_size + 1, output_dim).cuda())
        self.video_factor = Parameter(torch.Tensor(self.rank, in_size + 1, output_dim).cuda())
        self.text_factor = Parameter(torch.Tensor(self.rank, in_size + 1, output_dim).cuda())
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank).cuda())
        self.fusion_bias = Parameter(torch.Tensor(1, output_dim).cuda())

        xavier_normal(self.audio_factor)
        xavier_normal(self.video_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        xavier_normal(self.fusion_bias)

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, l1, a1, v1):
      #  a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
        DTYPE = torch.cuda.FloatTensor
        _audio_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), a1), dim=1)
        _video_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), v1), dim=1)
        _text_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), l1), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        y_2 = output.view(-1, self.output_dim)


        return y_2


class multiplication(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(multiplication, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
        xavier_normal(self.u1)

    def forward(self, l1, a1, v1):
     
      #  fusion = a1*v1
        fusion = a1*v1*l1        
      #  fusion = self.norm2(fusion)
     #   fusion = self.drop(fusion)
     #   y_1 = torch.relu(self.linear_1(fusion))
    #    y_1 = torch.relu(self.linear_2(y_1))
     #   y_2 = torch.relu(self.linear_3(y_1))
     #   y_3 = F.tanh(self.linear_3(y_2))

        return fusion


class concat(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(concat, self).__init__()
 

        self.linear_1 = nn.Linear(in_size*3, output_dim)
       # self.linear_1 = nn.Linear(in_size, hidden)


    def forward(self, l1, a1, v1):
     
        fusion = torch.cat([l1, a1, v1], dim=-1)

        y_1 = torch.relu(self.linear_1(fusion))


        return y_1



class addition(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(addition, self).__init__()
 

        self.linear_1 = nn.Linear(in_size*3, output_dim)
       # self.linear_1 = nn.Linear(in_size, hidden)


    def forward(self, l1, a1, v1): 
        y_1 = l1 + a1 + v1
        return y_1




