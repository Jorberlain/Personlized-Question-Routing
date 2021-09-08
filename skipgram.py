#Skip-Gram
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SkipGram(nn.Module):
    """
    Heterogeneous Entity Embedding Based Recommendation
    """
    def __init__(self
                 , embedding_dim
                 , emb_man
                 ):
        super(SkipGram, self).__init__()
        self.emb_dim = embedding_dim
        self.embedding_manager = emb_man

    def forward(self, rpos, apos, qinfo):
        emb = self.embedding_manager
        emb.zero_out()
        # R: 0, A: 1, Q: 2
        embed_ru = emb.ru_embeddings(rpos[0])
        embed_au = emb.au_embeddings(apos[0])

        embed_rv = emb.rv_embeddings(rpos[1])
        embed_av = emb.av_embeddings(apos[1])

        neg_embed_rv = emb.rv_embeddings(rpos[2])
        neg_embed_av = emb.av_embeddings(apos[2])

        quinput, qvinput, qninput = qinfo[:3]
        qulen, qvlen, qnlen = qinfo[3:]

        u_output, _ = emb.ubirnn(quinput, emb.init_hc(quinput.size(0)))
        v_output, _ = emb.vbirnn(qvinput, emb.init_hc(qvinput.size(0)))
        n_output, _ = emb.vbirnn(qninput, emb.init_hc(qninput.size(0)))

        u_pad = Variable(torch.zeros(u_output.size(0), 1, u_output.size(2)))
        v_pad = Variable(torch.zeros(v_output.size(0), 1, v_output.size(2)))
        n_pad = Variable(torch.zeros(n_output.size(0), 1, n_output.size(2)))

        if torch.cuda.is_available():
            u_pad = u_pad.cuda()
            v_pad = v_pad.cuda()
            n_pad = n_pad.cuda()

        u_output = torch.cat((u_pad, u_output), 1)
        v_output = torch.cat((v_pad, v_output), 1)
        n_output = torch.cat((n_pad, n_output), 1)

        qulen = qulen.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)
        qvlen = qvlen.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)
        qnlen = qnlen.unsqueeze(1).expand(-1, self.emb_dim).unsqueeze(1)

        embed_qu = u_output.gather(1, qulen.detach())
        embed_qv = v_output.gather(1, qvlen.detach())
        neg_embed_qv = n_output.gather(1, qnlen.detach())

        embed_u = embed_ru + embed_au + embed_qu.squeeze()
        embed_v = embed_rv + embed_av + embed_qv.squeeze()

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)

        log_sigmoid_pos = F.logsigmoid(score).squeeze()

        neg_embed_v = neg_embed_av + neg_embed_rv + neg_embed_qv.squeeze()
        neg_embed_v = neg_embed_v.view(quinput.size(0), -1, self.emb_dim)

        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        log_sigmoid_neg = F.logsigmoid(-1 * neg_score).squeeze()

        ne_loss = -1 * (log_sigmoid_pos + log_sigmoid_neg).sum()
        print("NE loss: {:.6f} ".format(ne_loss.data[0]), end=" ")

        return ne_loss
