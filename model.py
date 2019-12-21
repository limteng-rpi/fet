import math

import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo


class HFet(nn.Module):

    def __init__(self,
                 label_size,
                 elmo_option,
                 elmo_weight,
                 elmo_dropout=.5,
                 repr_dropout=.2,
                 dist_dropout=.5,
                 latent_size=0,
                 svd=None,
                 ):
        super(HFet, self).__init__()
        self.label_size = label_size
        self.elmo = Elmo(elmo_option, elmo_weight, 1,
                         dropout=elmo_dropout)
        self.elmo_dim = self.elmo.get_output_dim()

        self.attn_dim = 1
        self.attn_inner_dim = self.elmo_dim
        # Mention attention
        self.men_attn_linear_m = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.men_attn_linear_o = nn.Linear(self.attn_inner_dim, self.attn_dim, bias=False)
        # Context attention
        self.ctx_attn_linear_c = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_m = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_d = nn.Linear(1, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_o = nn.Linear(self.attn_inner_dim,
                                        self.attn_dim, bias=False)
        # Output linear layers
        self.repr_dropout = nn.Dropout(p=repr_dropout)
        self.output_linear = nn.Linear(self.elmo_dim * 2, label_size, bias=False)

        # SVD
        if svd:
            svd_mat = self.load_svd(svd)
            self.latent_size = svd_mat.size(1)
            self.latent_to_label.weight = nn.Parameter(svd_mat, requires_grad=True)
            self.latent_to_label.weight.requires_grad = False
        elif latent_size == 0:
            self.latent_size = int(math.sqrt(label_size))
        else:
            self.latent_size = latent_size
        self.latent_to_label = nn.Linear(self.latent_size, label_size,
                                         bias=False)
        self.latent_scalar = nn.Parameter(torch.FloatTensor([.1]))
        self.feat_to_latent = nn.Linear(self.elmo_dim * 2, self.latent_size,
                                        bias=False)
        # Loss function
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.mse = nn.MSELoss()
        # Relative position (distance)
        self.dist_dropout = nn.Dropout(p=dist_dropout)

    def load_svd(self, path):
        print('Loading SVD matrices')
        u_file = path + '-Ut'
        s_file = path + '-S'
        with open(s_file, 'r', encoding='utf-8') as r:
            s_num = int(r.readline().rstrip())
            mat_s = [[0] * s_num for _ in range(s_num)]
            for i in range(s_num):
                mat_s[i][i] = float(r.readline().rstrip())
        mat_s = torch.FloatTensor(mat_s)

        with open(u_file, 'r', encoding='utf-8') as r:
            mat_u = []
            r.readline()
            for line in r:
                mat_u.append([float(i) for i in line.rstrip().split()])
        mat_u = torch.FloatTensor(mat_u).transpose(0, 1)
        return torch.matmul(mat_u, mat_s) #.transpose(0, 1)

    def forward_nn(self, inputs, men_mask, ctx_mask, dist, gathers):
        # Elmo contextualized embeddings
        elmo_outputs = self.elmo(inputs)['elmo_representations'][0]
        _, seq_len, feat_dim = elmo_outputs.size()
        gathers = gathers.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, feat_dim)
        elmo_outputs = torch.gather(elmo_outputs, 0, gathers)

        men_attn = self.men_attn_linear_m(elmo_outputs).tanh()
        men_attn = self.men_attn_linear_o(men_attn)
        men_attn = men_attn + (1.0 - men_mask.unsqueeze(-1)) * -10000.0
        men_attn = men_attn.softmax(1)
        men_repr = (elmo_outputs * men_attn).sum(1)

        dist = self.dist_dropout(dist)
        ctx_attn = (self.ctx_attn_linear_c(elmo_outputs) +
                    self.ctx_attn_linear_m(men_repr.unsqueeze(1)) +
                    self.ctx_attn_linear_d(dist.unsqueeze(2))).tanh()
        ctx_attn = self.ctx_attn_linear_o(ctx_attn)

        ctx_attn = ctx_attn + (1.0 - ctx_mask.unsqueeze(-1)) * -10000.0
        ctx_attn = ctx_attn.softmax(1)
        ctx_repr = (elmo_outputs * ctx_attn).sum(1)

        # Classification
        final_repr = torch.cat([men_repr, ctx_repr], dim=1)
        final_repr = self.repr_dropout(final_repr)
        outputs = self.output_linear(final_repr)

        outputs_latent = None
        latent_label = self.feat_to_latent(final_repr) #.tanh()
        outputs_latent = self.latent_to_label(latent_label)
        outputs = outputs + self.latent_scalar * outputs_latent

        return outputs, outputs_latent

    def forward(self, inputs, labels, men_mask, ctx_mask, dist, gathers, inst_weights=None):
        outputs, outputs_latent = self.forward_nn(inputs, men_mask, ctx_mask, dist, gathers)
        loss = self.criterion(outputs, labels)
        return loss

    def _prediction(self, outputs, predict_top=True):
        _, highest = outputs.max(dim=1)
        highest = highest.int().tolist()
        preds = (outputs.sigmoid() > .5).int()
        if predict_top:
            for i, h in enumerate(highest):
                preds[i][h] = 1
        return preds

    def predict(self, inputs, men_mask, ctx_mask, dist, gathers, predict_top=True):
        self.eval()
        outputs, _ = self.forward_nn(inputs, men_mask, ctx_mask, dist, gathers)
        predictions = self._prediction(outputs, predict_top=predict_top)
        self.train()
        return predictions
