import gol
import numpy as np
import torch
import math
import random
import geoopt
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import softmax
import torch.nn.functional as F
import manifolds

from layers import SpaCn, BehCn, VP_SDE
import geoopt


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )
        for w in self.feed_forward:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        # x = self.dropout(self.norm1(attention + query))
        # forward = self.feed_forward(x)
        # out = self.dropout(self.norm2(forward + x))
        x = self.dropout(attention + query)
        forward = self.feed_forward(x)
        out = self.dropout(forward + x)
        return out

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        # self.embedding_layer = embedding_layer
        # self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(x)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out)

        return out

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the PE (positional encoding) with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Initialize a tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # Here is the content of sin and cos brackets, transformed through e and ln
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # Calculate PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Calculate PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # For convenience, unsqueeze to add a batch dimension
        pe = pe.unsqueeze(0)
        # If a parameter does not participate in gradient descent, but you want to save it when saving the model
        # You can use register_buffer in this case
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x is the inputs after embedding, for example (1, 7, 128), where the batch size is 1, there are 7 words, and the word dimension is 128
        """
        # Add positional encoding to x.
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(torch.float32).to(gol.device)
        self.relu = nn.ReLU().to(gol.device)
        self.fc2 = nn.Linear(hidden_dim, output_dim).to(torch.float32).to(gol.device)

    def forward(self, x):
        x = x.to(torch.float32).to(gol.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MPDC(nn.Module):
    def __init__(self, n_user, n_poi, spa_graph: Data):
        super(MPDC, self).__init__()
        self.n_user, self.n_poi = n_user, n_poi
        self.hid_dim = gol.conf['hidden']
        self.step_num = 1000
        self.local_pois = 20

        self.poi_emb = nn.Parameter(torch.empty(n_poi, self.hid_dim))
        self.distance_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        self.temporal_emb = nn.Parameter(torch.empty(gol.conf['interval'], self.hid_dim))
        nn.init.xavier_normal_(self.poi_emb)
        nn.init.xavier_normal_(self.distance_emb)
        nn.init.xavier_normal_(self.temporal_emb)

        self.spa_encoder = SpatialEncoder(n_poi, self.hid_dim, spa_graph).to(gol.device)
        self.beh_encoder = BehavirEncoder(self.hid_dim).to(gol.device)
        self.sde = VP_SDE(self.hid_dim, dt=gol.conf['dt']).to(gol.device)
        self.ce_criteria = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=1-gol.conf['keepprob'])
        self.ffn = MLP(128, 256, 128)

        self.beh_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8).to(gol.device)
        self.beh_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2).to(gol.device)
        self.score_fn = TransformerEncoder(embed_size=self.hid_dim,num_encoder_layers=1,num_heads=1,forward_expansion=2,dropout=0.2)


        self.spa_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8).to(gol.device)
        self.spa_attn_layernorm = nn.LayerNorm(self.hid_dim, eps=1e-8).to(gol.device)
        self.spa_attn = nn.MultiheadAttention(self.hid_dim, num_heads=2, batch_first=True, dropout=0.2).to(gol.device)

    def spa_condition(self,poi_embs, seqs):
        spa_embs = self.spa_encoder.encode(poi_embs)  # Distance Encoder
        if gol.conf['dropout']:
            spa_embs = self.dropout(spa_embs)
        spa_seq_embs = [spa_embs[seq] for seq in seqs]
        spa_embs_pad = pad_sequence(spa_seq_embs, batch_first=True, padding_value=0)

        return spa_embs_pad ,spa_embs

    def SpatialHGM(self, poi_embs, seqs, beh_encoder):  # Distance Encoder + Location Prototype

        beh_lengths = torch.LongTensor([seq.size(0) for seq in seqs]).to(gol.device)
        spa_embs_pad ,spa_embs = self.spa_condition(poi_embs,seqs)
        prompt = torch.nn.Parameter(torch.Tensor(1, self.hid_dim)).to(gol.device)
        torch.nn.init.xavier_uniform_(prompt).to(gol.device)
        beh_hid_embs = self.spa_layernorm(beh_encoder.detach().unsqueeze(1))
        pad_mask = beh_mask(beh_lengths)

        spa_embs_pad, _ = self.spa_attn(beh_hid_embs, spa_embs_pad, spa_embs_pad, key_padding_mask=~pad_mask)
        spa_embs_pad = spa_embs_pad.squeeze(1)
        spa_encoder = self.spa_attn_layernorm(spa_embs_pad)

        return spa_encoder, spa_embs

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def BehaviorGPM(self, poi_embs, beh_graph):  # Transition Encoder

        beh_embs = self.beh_encoder.encode((poi_embs, self.distance_emb, self.temporal_emb), beh_graph)
        if gol.conf['dropout']:
            beh_embs = self.dropout(beh_embs)
        ## Calculate the number of nodes in each batch from the beh_graph.batch tensor, beh_lengths will store the number of nodes or units corresponding to each batch
        beh_lengths = torch.bincount(beh_graph.batch)
        beh_embs = torch.split(beh_embs,
                               beh_lengths.cpu().numpy().tolist())  # len(beh_embs) 1024  beh_embs torch.Size([actual sequence length, 64])

        # Self-attention
        beh_embs_pad = pad_sequence(beh_embs, batch_first=True,
                                    padding_value=0)  # Pad to the longest sequence in this batch

        prompt = torch.nn.Parameter(torch.Tensor(1, self.hid_dim)).to(gol.device)
        torch.nn.init.xavier_uniform_(prompt).to(gol.device)
        prompt = prompt.expand(beh_embs_pad.size(0), -1, -1).to(gol.device)
        beh_hid_embs = self.beh_layernorm(beh_embs_pad)
        beh_embs_condition = self.ffn(beh_embs_pad)* prompt
        beh_embs_promt, _ = self.spa_attn(beh_hid_embs, beh_embs_condition , beh_embs_condition )
        pad_mask = beh_mask(beh_lengths)
        beh_embs_pad, _ = self.beh_attn(beh_hid_embs, beh_embs_pad, beh_embs_pad, key_padding_mask=~pad_mask)
        beh_embs_pad = beh_embs_pad + beh_hid_embs
        beh_embs_pad = [seq[:seq_len] for seq, seq_len in zip(beh_embs_pad, beh_lengths)]
        beh_encoder = torch.stack([seq.mean(dim=0) for seq in beh_embs_pad], dim=0)

        beh_embs_pad_prompt, _ = self.beh_attn(beh_hid_embs , beh_embs_promt,
                                            beh_embs_promt, key_padding_mask=~pad_mask)
        beh_embs_pad_prompt = beh_embs_pad_prompt + beh_hid_embs

        beh_embs_pad_prompt = [seq[:seq_len] for seq, seq_len in zip(beh_embs_pad_prompt, beh_lengths)]
        beh_encoder_prompt = torch.stack([seq.mean(dim=0) for seq in beh_embs_pad_prompt], dim=0)

        return beh_encoder, beh_embs, beh_encoder_prompt

    def sde_sample(self, spa_encoder, beh_encoder, target=None):
        local_embs = spa_encoder
        condition_embs = beh_encoder.detach()
        sde_encoder = self.sde.reverse_sde(local_embs, condition_embs, gol.conf['T'])

        fisher_loss = None
        if target is not None: # training phase
            t_sampled = np.random.randint(1, self.step_num) / self.step_num
            mean, std = self.sde.marginal_prob(target, t_sampled)
            z = torch.randn_like(target)
            perturbed_data = mean + std.unsqueeze(-1) * z
            score = - self.sde.calc_score(perturbed_data, condition_embs)
            fisher_loss = torch.square(score + z).mean()

        return sde_encoder, fisher_loss

    def getTrainLoss(self, batch):
        usr, pos_lbl, _, seqs, beh_graph, cur_time = batch
        poi_embs = self.poi_emb
        if gol.conf['dropout']:
            poi_embs = self.dropout(poi_embs)
        beh_encoder, beh_embs, beh_encoder_prompt = self.BehaviorGPM(poi_embs, beh_graph)
        spa_encoder, spa_embs = self.SpatialHGM(poi_embs, seqs, beh_encoder_prompt)
        sde_encoder, fisher_loss = self.sde_sample(spa_encoder, beh_encoder, target=spa_embs[pos_lbl])
        sde_encoder_prompt, fisher_loss2 = self.sde_sample(spa_encoder, beh_encoder_prompt, target=spa_embs[pos_lbl])
        cl_loss = self.InfoNCE(sde_encoder_prompt, sde_encoder, 0.2)
        fisher_loss = 0.5*fisher_loss + 0.5*fisher_loss2
        pred_logits = beh_encoder_prompt @ self.poi_emb.T + spa_encoder @ spa_embs.T
        # pred_logits = 0.7*(beh_encoder @ self.poi_emb.T) + 0.3*(sde_encoder @ spa_embs.T)
        return self.ce_criteria(pred_logits, pos_lbl), fisher_loss, cl_loss


    def forward(self, seqs, beh_graph):
        poi_embs = self.poi_emb
        beh_encoder, beh_embs, beh_encoder_prompt = self.BehaviorGPM(poi_embs, beh_graph)
        # beh_encoder,fisher_seq = self.sdeProp(beh_encoder,beh_encoder_prompt,target=beh_encoder)
        spa_encoder, spa_embs = self.SpatialHGM(poi_embs, seqs, beh_encoder_prompt)
        # sde_encoder, fisher_loss = self.sde_sample(spa_encoder, beh_encoder)
        sde_encoder_prompt, fisher_loss2 = self.sde_sample(spa_encoder, beh_encoder_prompt)

        pred_logits = beh_encoder_prompt @ self.poi_emb.T + spa_encoder @ spa_embs.T
        return pred_logits

class BehavirEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(BehavirEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.encoder = BehCn(hid_dim)

    def encode(self, embs, beh_graph):
        return self.encoder(embs, beh_graph)

class SpatialEncoder(nn.Module):
    def __init__(self, n_poi, hid_dim, spa_graph: Data):
        super(SpatialEncoder, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.gcn_num = gol.conf['num_layer']

        edge_index, _ = add_self_loops(spa_graph.edge_index)
        dist_vec = torch.cat([spa_graph.edge_attr, torch.zeros((n_poi,)).to(gol.device)])
        dist_vec = torch.exp(-(dist_vec ** 2))
        self.spa_graph = Data(edge_index=edge_index, edge_attr=dist_vec)
        self.c = torch.tensor(5)
        self.manifold = getattr(manifolds, "Hyperboloid")()
        # self.lorentz = geoopt.manifolds.PoincareBall()
        self.lorentz = geoopt.manifolds.Lorentz()

        self.act = nn.LeakyReLU()
        self.spa_convs = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.spa_convs.append(SpaCn(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        layer_embs = self.manifold.proj(self.manifold.expmap0(layer_embs, c=self.c), c=self.c)
        layer_embs = self.lorentz.logmap0(layer_embs)
        spa_embs = [layer_embs]
        for conv in self.spa_convs:
            layer_embs = conv(layer_embs, self.spa_graph)
            layer_embs = self.act(layer_embs)
            spa_embs.append(layer_embs)
        spa_embs = torch.stack(spa_embs, dim=1).mean(1)
        return spa_embs

def beh_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)

