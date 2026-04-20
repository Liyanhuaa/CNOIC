from util import *

class BertForModel(BertPreTrainedModel):
    def __init__(self,config,num_labels):  #config通常是一个BERT模型的配置对象，而num_labels表示要分类的标签数量
        super(BertForModel, self).__init__(config) #调用了父类BertPreTrainedModel的构造函数，并传递了config参数
        self.num_labels = num_labels
        self.bert = BertModel(config) #BERT模型 被创建并存储在 self.bert 变量中
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) #创建了一个线性层，将其存储在 self.dense中
        #nn.Linear 是PyTorch中的线性层构造函数，它接受两个参数，分别是输入特征的大小和输出特征的大小
        self.activation = nn.ReLU() #定义了ReLU（修正线性单元）激活函数，它将用于非线性变换的激活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #nn.Dropout 是PyTorch中的丢弃层构造函数，它接受一个参数，即丢弃的概率,减少模型的过拟合风险
        self.classifier = nn.Linear(config.hidden_size,num_labels) #用于执行分类任务的输出层，它将模型的隐藏表示映射到类别空间
        self.feature_transform = nn.Linear(config.hidden_size, 768)######这个是我增加的
        self.apply(self.init_bert_weights) #用了 self.init_bert_weights 函数对模型的权重进行初始化，并将初始化应用于模型的各个组件

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        #encoded_layer_12 包含了BERT的所有编码层的输出，而 pooled_output 包含了经过汇总的模型输出
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1)) #这行代码对BERT的最后一个编码层（encoded_layer_12[-1]）进行平均汇总,然后通过一个线性层 self.dense 进行变换
        pooled_output = self.activation(pooled_output) #将线性变换后的输出传递给激活函数 self.activation
        pooled_output = self.dropout(pooled_output) #对前一步的输出进行丢弃操作，以减少过拟合风险
        logits = self.classifier(pooled_output) #将处理后的特征映射到类别空间，生成模型的输出 logits
        
        if feature_ext:
            pooled_output = self.feature_transform(pooled_output)  ##这一行也是我增加的
            return pooled_output
        else:
            if mode == 'train':
                loss = nn.CrossEntropyLoss()(logits,labels)
                return loss
            else:
                pooled_output = self.feature_transform(pooled_output)  ##这一行也是我增加的这一行也是我加的
                return pooled_output, logits
                    

