import torch
import torch.nn as nn
import torch.nn.functional as F
'''
自动学习图中每个节点应该给予其不同邻居多少重要性（权重）
s
'''



class GraphAttentionLayer(nn.Module):
    def __init__(self,in_feature,out_feature,dropout,aplha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.dropout=dropout
        self.alpha=aplha
        self.concat=concat

        self.Wlinear=nn.Linear(in_feature,out_feature)
        # self.W=nn.Parameter(torch.empty(size=(batch_size,in_feature,out_feature)))
        nn.init.xavier_uniform_(self.Wlinear.weight,gain=1.414)

        self.aiLinear=nn.Linear(out_feature,1)
        self.ajLinear=nn.Linear(out_feature,1)
        # self.a=nn.Parameter(torch.empty(size=(batch_size,2*out_feature,1)))
        nn.init.xavier_uniform_(self.aiLinear.weight,gain=1.414)
        nn.init.xavier_uniform_(self.ajLinear.weight,gain=1.414)

        self.leakyRelu=nn.LeakyReLU(self.alpha)

    # 计算图中任意两个节点 $i$ 和 $j$ 之间特征的匹配程度，作为后续 Softmax 归一化前的权重。
    def getAttentionE(self,Wh):
        #重点改了这个函数
        # Wh1 的形状变由 (Batch Size, N, F)变为 (Batch Size, N, 1)
        Wh1=self.aiLinear(Wh)
        Wh2=self.ajLinear(Wh)
        # 将 Wh2 的形状从 (B, N, 1) 转换为 (B, 1, N)。
        Wh2=Wh2.view(Wh2.shape[0],Wh2.shape[2],Wh2.shape[1])
        # Wh1=torch.bmm(Wh,self.a[:,:self.out_feature,:])    #Wh:size(node,out_feature),a[:out_eature,:]:size(out_feature,1) => Wh1:size(node,1)
        # Wh2=torch.bmm(Wh,self.a[:,self.out_feature:,:])    #Wh:size(node,out_feature),a[out_eature:,:]:size(out_feature,1) => Wh2:size(node,1)

        # 矩阵e中的eij元素就是节点 i对节点 j的原始注意力分数
        e=Wh1+Wh2   #broadcast add, => e:size(node,node)
        return self.leakyRelu(e)

    def forward(self,h,adj):
        # print(h.shape)
        Wh=self.Wlinear(h)
        # Wh=torch.bmm(h,self.W)   #h:size(node,in_feature),W:size(in_feature,out_feature) => Wh:size(node,out_feature)
        e=self.getAttentionE(Wh)

        zero_vec=-1e9*torch.ones_like(e)
        attention=torch.where(adj>0,e,zero_vec)
        attention=F.softmax(attention,dim=2)
        attention=F.dropout(attention,self.dropout,training=self.training)
        h_hat=torch.bmm(attention,Wh)  #attention:size(node,node),Wh:size(node,out_fature) => h_hat:size(node,out_feature)

        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

    def __repr__(self):
        return self.__class__.__name__+' ('+str(self.in_feature)+'->'+str(self.out_feature)+')'


class GAT(nn.Module):
    def __init__(self,in_feature,hidden_feature,out_feature,attention_layers,dropout,alpha):
        super(GAT,self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.hidden_feature=hidden_feature
        self.dropout=dropout
        self.alpha=alpha
        self.attention_layers=attention_layers

        self.attentions=[GraphAttentionLayer(in_feature,hidden_feature,dropout,alpha,True) for i in range(attention_layers)]

        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

        self.out_attention=GraphAttentionLayer(attention_layers*hidden_feature,out_feature,dropout,alpha,False)


    def forward(self,h,adj):
        # print(h)
        h=F.dropout(h,self.dropout,training=self.training)

        h=torch.cat([attention(h,adj) for attention in self.attentions],dim=2)
        h=F.dropout(h,self.dropout,training=self.training)
        h=F.elu(self.out_attention(h,adj))
        return h

