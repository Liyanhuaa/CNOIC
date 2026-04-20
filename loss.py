from util import *
import torch
def euclidean_metric(a, b):
    n = a.shape[0] #获取张量a的第一个维度大小，通常表示样本数
    m = b.shape[0] #获取张量b的第一个维度大小，通常表示样本数
    a = a.unsqueeze(1).expand(n, m, -1)
    #将张量a的维度扩展，使其具有三个维度。首先，使用unsqueeze(1)将维度扩展到第二个维度上，然后使用expand来将其复制成(n, m, -1)的形状，其中-1表示该维度的大小由张量自动计算
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class BoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):  #这个损失函数的主要目的可能是在训练期间优化参数 self.delta 以最小化损失函数的值
        
        super(BoundaryLoss, self).__init__() #调用了父类 nn.Module 的构造函数，确保损失函数的正确初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.feat_dim = feat_dim #表示特征维度的数量
        self.delta = nn.Parameter(torch.randn(num_labels).cuda()) #.cuda() 方法，将 self.delta 放置在 GPU 上
        nn.init.normal_(self.delta) #这一行对 self.delta 进行了正态分布的初始化，以确保参数具有适当的初始值
        self.feat_norm = nn.LayerNorm(feat_dim).to(self.device)
    def l2_normalize(self, x):
            return F.normalize(x, p=2, dim=-1)

    def forward(self, pooled_output, centroids, labels):

        #pooled_output = self.feat_norm(pooled_output)
        #pooled_output = self.l2_normalize(pooled_output)
        
        logits = euclidean_metric(pooled_output, centroids)
        #logits = torch.norm(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1) 
        delta = F.softplus(self.delta)  #激活函数
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output
        
        euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask  #为了求和
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()
        
        return loss, delta 