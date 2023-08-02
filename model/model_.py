import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        # Conv2d requires shape = (num_sample, c_in, heigh, width) or (c_in, height, width)
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1) # shape = (num_sample, var, dim, num_his)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1) # shape = (num_sample, num_his?, dim?, D)

class conv3d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super().__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0, 0]
        self.conv = nn.Conv3d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm3d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1) 
        x = F.pad(x, ([self.padding_size[2], self.padding_size[2], self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 4, 2, 3, 1) # shape = (num_sample, num_his?, dim?, D)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, expand=False, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        if not expand:
            self.convs = nn.ModuleList([conv2d_(
                input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
                padding='VALID', use_bias=use_bias, activation=activation,
                bn_decay=bn_decay) for input_dim, num_unit, activation in
                zip(input_dims, units, activations)])
        else: 
            self.convs = nn.ModuleList([conv3d_(
                input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1, 1], stride=[1, 1, 1],
                padding='VALID', use_bias=use_bias, activation=activation,
                bn_decay=bn_decay) for input_dim, num_unit, activation in
                zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STPEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    PE:     [num_var, 2]
    T:      num of time steps in one day
    D:      output dims
    return: [batch_size, num_his + num_pred, num_vertex, num_var, D]
    '''

    def __init__(self, D, T, bn_decay):
        super().__init__()
        self.FC_pe = FC(
            input_dims=[2, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, expand=True)
        
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, expand=True)

        self.FC_te = FC(
            input_dims=[T+7, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, expand=True)  

    def forward(self, SE, TE, T):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0).unsqueeze(3) # shape = (1, 1, num_vertex/dim, 1, D) 
        SE = self.FC_se(SE)
        # temporal embedding shape: (batch_size, num_step, 7) & (batch_size, num_step, T)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2).unsqueeze(3)
        TE = self.FC_te(TE)
        # physical embedding
        PE = F.one_hot(torch.arange(0, 2))
        PE = PE.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(TE.device, dtype = torch.float)
        PE = self.FC_pe(PE)
        del dayofweek, timeofday
        return SE + TE + PE

class physicalAttention(nn.Module):
    '''
    physical attention mechanism
    X:      [batch_size, num_step, num_vertex, num_var, D]
    STE:    [batch_size, num_step, num_vertex, num_var, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, num_var, D]
    '''

    def __init__(self, K, d, bn_decay):
        super().__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, expand=True)

    def forward(self, X, STPE):
        batch_size = X.shape[0]
        X = torch.cat((X, STPE), dim=-1)
        # [batch_size, num_step, num_vertex, num_var, k * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, num_var, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_var, num_var]
        attention = torch.matmul(query, key.transpose(3, 4))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X



class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, num_var, D]
    STE:    [batch_size, num_step, num_vertex, num_var, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, num_var, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, expand=True)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, num_var, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, num_var, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        query = query.permute(0, 1, 3, 2, 4)
        key = key.permute(0, 1, 3, 4, 2)
        value = value.permute(0, 1, 3, 2, 4)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 1, 3, 2, 4)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, num_var, D]
    STE:    [batch_size, num_step, num_vertex, num_var, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, num_var, D]
    '''

    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, expand=True)

    def forward(self, X, STPE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STPE), dim=-1)
        # [batch_size, num_step, num_vertex, num_var, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_var, num_vertex, num_step, d]
        # key:   [K * batch_size, num_var, num_vertex, d, num_step]
        # value: [K * batch_size, num_var, num_vertex, num_step, d]
        query = query.permute(0, 3, 2, 1, 4)
        key = key.permute(0, 3, 2, 4, 1)
        value = value.permute(0, 3, 2, 1, 4)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 3, 2, 1, 4)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, num_var, D]
    HT:     [batch_size, num_step, num_vertex, num_var, D]
    HP:     [batch_size, num_step, num_vertex, num_var, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, num_var, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, expand=True, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, expand=True, use_bias=True)
        self.FC_xp = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, expand=True, use_bias=False)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay, expand=True)

    def forward(self, HS, HT, HP):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        XP = self.FC_xp(HP)
        x = torch.sigmoid(torch.add(XS, XT))
        y = torch.sigmoid(torch.add(XS, XP))
        z = torch.sigmoid(torch.add(XT, XP))
        H = torch.add(torch.add(torch.mul(x+y, HS), torch.mul(x+z, HT)), torch.mul(y+z, HT))
        H = self.FC_h(H)
        del XS, XT, XP, x, y, z
        return H


class STPAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        super().__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.physicalAttention = physicalAttention(K, d, bn_decay)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STPE):
        HS = self.spatialAttention(X, STPE)
        HT = self.temporalAttention(X, STPE)
        HP = self.physicalAttention(X, STPE)
        H = self.gatedFusion(HS, HT, HP)
        del HS, HT, HP
        return torch.add(X, H)


class transformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, num_var, D]
    STE_his:  [batch_size, num_his, num_vertex, num_var, D]
    STE_pred: [batch_size, num_pred, num_vertex, num_var, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, num_var, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay, expand=True)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay, expand=True)

    def forward(self, X, STPE_his, STPE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STPE_pred)
        key = self.FC_k(STPE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_var, num_pred, d]
        # key:   [K * batch_size, num_vertex, num_var, d, num_his]
        # value: [K * batch_size, num_vertex, num_var, num_his, d]
        query = query.permute(0, 3, 2, 1, 4)
        key = key.permute(0, 3, 2, 4, 1)
        value = value.permute(0, 3, 2, 1, 4)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 3, 2, 1, 4)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    '''
    GMAN
        X：       [batch_size, num_his, num_vertx, num_var]
        TE：      [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE：      [num_vertex, K * d]
        num_his： number of history steps
        num_pred：number of prediction steps
        T：       one day is divided into T steps
        L：       number of STAtt blocks in the encoder/decoder
        K：       number of attention heads
        d：       dimension of each attention head outputs
        return：  [batch_size, num_pred, num_vertex, num_var]
    '''

    def __init__(self, SE, args, bn_decay):
        super(GMAN, self).__init__()
        T = 24 * 60 // args.time_slot
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        self.num_his = args.num_his
        self.SE = SE
        self.STPEmbedding = STPEmbedding(D, T, bn_decay)
        self.STPAttBlock_1 = nn.ModuleList([STPAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STPAttBlock_2 = nn.ModuleList([STPAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay, expand = True)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay, expand = True)

    def forward(self, X, TE):

        # input
        X = torch.unsqueeze(X, -1) # shape = (num_sample, num_his, dim, var, 1)
        X = self.FC_1(X)
        # STE
        STPE = self.STPEmbedding(self.SE, TE, self.T)
        STPE_his = STPE[:, :self.num_his]
        STPE_pred = STPE[:, self.num_his:]
        # encoder
        for net in self.STPAttBlock_1:
            X = net(X, STPE_his)
        # transAtt
        X = self.transformAttention(X, STPE_his, STPE_pred)
        # decoder
        for net in self.STPAttBlock_2:
            X = net(X, STPE_pred)
        # output
        X = self.FC_2(X)
        del STPE, STPE_his, STPE_pred
        return torch.squeeze(X, -1) # shape = (num_sample, num_his, dim, var)
