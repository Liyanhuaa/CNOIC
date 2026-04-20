from util import *
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from transformers import BertTokenizer
import csv
import torch
import pickle
import os


class Data:

    def __init__(self, args):
        set_seed(args.seed)
        max_seq_lengths = {'oos': 30, 'clinc': 32, 'dbpedia': 30, 'stackoverflow': 45, 'banking': 55, 'snips': 35,
                           'ATIS': 55,'tc20':500}  # 字典，包含了不同数据集的最大序列长度。根据传入的args.dataset参数来选择适当的最大序列长度
#        max_seq_lengths = {'oos': 30, 'clinc': 32, 'dbpedia': 54, 'stackoverflow': 45, 'banking': 55, 'snips': 35,
#                           'ATIS': 55,'tc20':500}  # 字典，包含了不同数据集的最大序列长度。根据传入的args.dataset参数来选择适当的最大序列长度
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.open_noise_data_dir = os.path.join(args.data_dir, args.open_noise_dataset)

        self.all_label_list = processor.get_labels(self.data_dir)  # 包含所有标签的列表
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)  # round 四舍五入  表示已知类别的数量
        set_seed(args.seed)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.n_open_noise_cls = round(len(self.all_label_list)* (1-args.known_cls_ratio)/2)  #ijcai加的  意思是把原来的样本中剩余的一般当做是未知类、一般当做是开放噪声类
        set_seed(args.seed)
        self.n_open_noise_cls_list = list(np.random.choice(np.array(list(set(self.all_label_list) - set(self.known_label_list))), self.n_open_noise_cls, replace=False))
        self.unknown_label_list = list(set(self.all_label_list) - set(self.known_label_list)-set( self.n_open_noise_cls_list))


        self.num_labels = len(self.known_label_list)

        if args.dataset == 'oos':
            self.unseen_token = 'oos'
        else:
            self.unseen_token = '<UNK>'
        self.ood_token = '<OOD>'

        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list +[self.unseen_token]#已知类+未知类
        self.label_list_includeood=self.label_list+[self.ood_token]

        self.three_label_list = self.known_label_list +self.n_open_noise_cls_list+ [self.unseen_token]#已知类+开放噪声类+未知类
        self.train_examples = self.get_examples_noise(processor, args, 'train')
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')

        self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)
        set_seed(args.seed)
        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    # 检查当前示例的标签 example.label 是否属于已知标签列表 self.known_label_list
                    example.noise_label = example.label
                    examples.append(example)
        elif mode == 'eval':
            for example in ori_examples:
                if example.label in self.known_label_list:
                    example.noise_label=example.label
                    examples.append(example)
            save_path = f"samples_{args.dataset}_{args.known_cls_ratio}_{args.open_noise_dataset}_{args.ind_noise_ratio}_{args.ood_noise_ratio}_{mode}.csv"
            label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
            with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                for example in examples:
                    numeric_noise_label = label_to_id.get(example.noise_label, -1)  # 如果未找到，设置为 -1
                    writer.writerow([numeric_noise_label, example.text_a])
            print(f"Samples saved to {save_path}")
        elif mode == 'test':
            for example in ori_examples:
                if example.label in self.known_label_list:
                    example.noise_label = example.label
                    examples.append(example)
                if example.label in self.unknown_label_list:
                    example.label = self.unseen_token
                    example.noise_label = self.unseen_token
                    examples.append(example)
            save_path = f"samples_{args.dataset}_{args.known_cls_ratio}_{args.open_noise_dataset}_{args.ind_noise_ratio}_{args.ood_noise_ratio}_{mode}.csv"
            label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
            with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                #writer.writerow(["text",  "true_label","noise_label"])  # 写入表头
                #writer.writerow([ "noise_label","text"])  # 写入表头

                for example in examples:
                    numeric_noise_label = label_to_id.get(example.noise_label, -1)  # 如果未找到，设置为 -1
                    writer.writerow([numeric_noise_label, example.text_a])
                    #writer.writerow([example.text_a, example.label,example.noise_label])


            print(f"Samples saved to {save_path}")
        return examples



    def get_examples_noise(self, processor, args, mode='train'):
        # 设置固定的随机种子
        set_seed(args.seed)

        ori_examples = processor.get_examples(self.data_dir, mode)

        if args.ood_type == 'near':
            set_seed(args.seed)

            ind_examples = []
            ood_examples = []

            # 拆分样本
            known_examples = [e for e in ori_examples if e.label in self.known_label_list]
            ood_candidates = [e for e in ori_examples if e.label in self.n_open_noise_cls_list]

            num_known_samples = len(known_examples)
            num_ood_samples = len(ood_candidates)

            num_ind_select = int(num_known_samples * args.ind_noise_ratio / (1 - args.ood_noise_ratio))
            num_ood_select = int(num_known_samples * args.ood_noise_ratio / (1 - args.ood_noise_ratio))

            # 随机选 IND 噪声索引
            ind_noise_indices = np.random.choice(num_known_samples, size=num_ind_select, replace=False)
            for idx, example in enumerate(known_examples):
                if idx in ind_noise_indices:
                    possible_labels = [label for label in self.known_label_list if label != example.label]
                    example.noise_label = random.choice(possible_labels)
                else:
                    example.noise_label = example.label
                ind_examples.append(example)

            # 随机选 OOD 噪声索引
            ood_noise_indices = np.random.choice(len(ood_candidates), size=num_ood_select, replace=False)
            for idx, example in enumerate(ood_candidates):
                if idx in ood_noise_indices:
                    example.noise_label = random.choice(self.known_label_list)
                    example.label = self.ood_token
                    ood_examples.append(example)

            # 最终样本
            examples = ind_examples + ood_examples

        if args.ood_type == 'far':
            set_seed(args.seed)

            ind_examples = []
            ood_noise_examples = []

            ori_noise_examples = processor.get_examples(self.open_noise_data_dir, mode)

            # 拆分
            known_examples = [e for e in ori_examples if e.label in self.known_label_list]
            num_known_samples = len(known_examples)
            num_open_noise_samples = len(ori_noise_examples)

            num_ind_select = int(num_known_samples * args.ind_noise_ratio / (1 - args.ood_noise_ratio))
            num_ood_select = int(num_known_samples * args.ood_noise_ratio / (1 - args.ood_noise_ratio))

            # IND 噪声注入
            ind_noise_indices = np.random.choice(num_known_samples, size=num_ind_select, replace=False)
            for idx, example in enumerate(known_examples):
                if idx in ind_noise_indices:
                    possible_labels = [label for label in self.known_label_list if label != example.label]
                    example.noise_label = random.choice(possible_labels)
                else:
                    example.noise_label = example.label
                ind_examples.append(example)

            # OOD 噪声注入
            ood_noise_indices = np.random.choice(num_open_noise_samples, size=num_ood_select, replace=False)
            for idx, example in enumerate(ori_noise_examples):
                if idx in ood_noise_indices:
                    example.noise_label = random.choice(self.known_label_list)
                    example.label = self.ood_token
                    ood_noise_examples.append(example)

            examples = ind_examples + ood_noise_examples



        return examples

    def get_loader(self, examples, args, mode='train'):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(examples, self.label_list_includeood, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_noiseids = torch.tensor([f.label_noiseid for f in features], dtype=torch.long)
        indices = torch.arange(len(features), dtype=torch.long)
        datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_noiseids, indices)

        if mode == 'train':
            # ========== 关键修复：确保 DataLoader 完全确定性 ==========
            set_seed(args.seed)

            # 创建 generator 并设置种子
            g = torch.Generator()
            g.manual_seed(args.seed)

            # 使用 generator 创建 sampler
            sampler = RandomSampler(datatensor, generator=g)

            # 关键：DataLoader 也要使用同一个 generator！
            dataloader = DataLoader(
                datatensor,
                sampler=sampler,
                batch_size=args.train_batch_size,
                drop_last=False,
                num_workers=0,
                generator=g  # ← 关键：DataLoader 也要用同一个 generator！
            )
            # =========================================================

        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.eval_batch_size, num_workers=0)

        return dataloader
    def get_loader_(self, examples, args, mode='train'):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(examples, self.label_list_includeood, args.max_seq_length, tokenizer)#self.three_label_list指的是包含 已知 噪声和unknown的标签n+n+1
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_noiseids = torch.tensor([f.label_noiseid for f in features], dtype=torch.long)#这个是自己增加的样本噪声
        indices = torch.arange(len(features), dtype=torch.long)#这个是自己增加的索引
        datatensor = TensorDataset(input_ids, input_mask, segment_ids, label_ids,label_noiseids,indices)

        if mode == 'train':
            set_seed(args.seed)
            g = torch.Generator()
            g.manual_seed(args.seed)
            sampler = RandomSampler(datatensor, generator=g)
            # sampler = RandomSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.train_batch_size,drop_last=False,num_workers=0)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.eval_batch_size,num_workers=0)

        return dataloader



class InputExample(object):  # 创建一组 InputExample 对象，每个对象表示一个训练样本或测试样本，然后将它们传递给模型进行训练或评估
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,label_noiseid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_noiseid=label_noiseid


class DataProcessor(object):  #
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:  # 指定使用 UTF-8 编码
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)  # 读取tsv文件并解析为多行数据的列表
            lines = []
            for line in reader:
                lines.append(line)
            return lines



class DatasetProcessor(DataProcessor):  # DatasetProcessor 类将继承 DataProcessor 类中定义的方法和属性

    def get_examples(self, data_dir, mode):  # data_dir 表示数据目录的路径，mode 表示处理的模式
        if mode == 'train':
            return self._create_examples(  # self._create_examples(...) 方法用于将上一步的列表数据转换为示例（InputExample）的集合
                self._read_tsv(os.path.join(data_dir, "train.tsv")),
                "train")  ##会读取数据目录中的 "train.tsv" 文件，并将其内容解析为一个列表的列表
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):  # 从指定的数据文件中读取标签信息，并返回一个包含了数据集中所有可能标签的列表
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        # test = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=",")  #从指定的 "train.tsv" 文件中读取数据，数据以制表符分隔
        labels = np.unique(np.array(test[
                                        'label']))  # 从DataFrame中选择名为 'label' 的列，该列包含了示例的标签。然后，使用 np.unique 函数从该列中提取所有唯一的标签值，并将其存储在名为 labels 的NumPy数组中

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:  # 条件判断，它跳过了列表中的第一行，通常是表头或列名
                continue
            if len(line) != 2:  # 检查当前行的长度是否等于 2。如果不等于 2，说明这行数据不符合预期的格式，可能是不完整的数据或格式错误的数据，因此将跳过
                continue
            guid = "%s-%s" % (set_type, i)  # 根据示例集合的类型和索引 i 创建一个唯一的示例标识符 guid
            text_a = line[0]  # 提取当前行的第一个元素，通常是文本序列，将其存储在 text_a
            label = line[1]  # label = line[1]: 提取当前行的第二个元素，通常是标签信息，将其存储在 label 中

            examples.append(
                # 使用提取的信息创建一个 InputExample 对象，并将其添加到 examples 列表中。这个对象包括了示例的唯一标识符 guid、文本序列 text_a 和标签 label
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):  # 将示例集合转换为特征集合，tokenizer: 这是一个用于将文本转换为模型输入的分词器
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}  # 空字典，用于创建标签到整数索引的映射
    for i, label in enumerate(label_list):  # 这段代码的主要作用是创建一个标签到整数索引的映射
        label_map[label] = i  # 将文本标签映射为整数标签

    features = []
    for (ex_index, example) in enumerate(examples):  # ex_index 是示例在列表中的索引，example 是示例对象。
        tokens_a = tokenizer.tokenize(
            example.text_a)  # 对示例对象 example 的文本序列 text_a 进行分词，使用了提供的 tokenizer。分词是将文本分割成单词或子词的过程

        # 这段代码的目的是确保文本序列的长度不超过模型的最大输入长度要求
        tokens_b = None  # 初始化一个变量 tokens_b 为 None，以用于存储文本序列 text_b 的分词结果
        if example.text_b:  # 如果示例对象中存在 text_b
            tokens_b = tokenizer.tokenize(example.text_b)

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            # truncate_seq_pair 函数，该函数用于修改 tokens_a 和 tokens_b，以确保它们的总长度不超过指定的 max_seq_length - 3。这个操作通常包括考虑特殊标志 [CLS]、[SEP] 和 [SEP] 的长度
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]  # 将分词后的文本序列 tokens_a 转换为一个新的列表 tokens
        segment_ids = [0] * len(tokens)  # 创建一个与文本序列 tokens 相对应的 segment_ids 列表，并将所有标记都标识为同一个文本段落0

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(
            tokens)  # 分词器 tokenizer 将文本序列 tokens 中的标记（通常是单词或子词）转换为对应的整数标识符 input_ids

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)  # 没有填充

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))  # 对输入序列进行填充，以确保其长度达到指定的 max_seq_length
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length  # 检查以确保填充后的输入序列长度和max_seq_length的长度相等
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        label_noiseid=label_map[example.noise_label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          label_noiseid=label_noiseid
                          ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):  # 定义函数用于截断一对文本序列 tokens_a 和 tokens_b，使它们的总长度不超过指定的 max_length
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context  #tokens_a 的开头（左侧）移除一个标记。这个操作表示从较长的序列中截断一个标记，以减少总长度
        else:
            tokens_b.pop()  # 从 tokens_b 的末尾（右侧）移除一个标记。这个操作表示从较长的序列中截断一个标记，以减少总长度

def save_dataloader_index(indices, save_path):
    dir_path = os.path.dirname(save_path)
    if dir_path:  # 防止空路径报错
        os.makedirs(dir_path, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(indices, f)
    print(f"[Info] Dataloader indices saved to: {save_path}")

def load_dataloader_index(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"[Error] Index file not found: {load_path}")
    with open(load_path, 'rb') as f:
        indices = pickle.load(f)
    print(f"[Info] Dataloader indices loaded from: {load_path}")
    return indices