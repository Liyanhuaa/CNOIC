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
                           'ATIS': 55,'tc20':500}

        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.open_noise_data_dir = os.path.join(args.data_dir, args.open_noise_dataset)

        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        set_seed(args.seed)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.n_open_noise_cls = round(len(self.all_label_list)* (1-args.known_cls_ratio)/2)
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
        self.label_list = self.known_label_list +[self.unseen_token]
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
                    numeric_noise_label = label_to_id.get(example.noise_label, -1)
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


                for example in examples:
                    numeric_noise_label = label_to_id.get(example.noise_label, -1)
                    writer.writerow([numeric_noise_label, example.text_a])



            print(f"Samples saved to {save_path}")
        return examples



    def get_examples_noise(self, processor, args, mode='train'):

        set_seed(args.seed)

        ori_examples = processor.get_examples(self.data_dir, mode)

        if args.ood_type == 'near':
            set_seed(args.seed)

            ind_examples = []
            ood_examples = []


            known_examples = [e for e in ori_examples if e.label in self.known_label_list]
            ood_candidates = [e for e in ori_examples if e.label in self.n_open_noise_cls_list]

            num_known_samples = len(known_examples)
            num_ood_samples = len(ood_candidates)

            num_ind_select = int(num_known_samples * args.ind_noise_ratio / (1 - args.ood_noise_ratio))
            num_ood_select = int(num_known_samples * args.ood_noise_ratio / (1 - args.ood_noise_ratio))


            ind_noise_indices = np.random.choice(num_known_samples, size=num_ind_select, replace=False)
            for idx, example in enumerate(known_examples):
                if idx in ind_noise_indices:
                    possible_labels = [label for label in self.known_label_list if label != example.label]
                    example.noise_label = random.choice(possible_labels)
                else:
                    example.noise_label = example.label
                ind_examples.append(example)


            ood_noise_indices = np.random.choice(len(ood_candidates), size=num_ood_select, replace=False)
            for idx, example in enumerate(ood_candidates):
                if idx in ood_noise_indices:
                    example.noise_label = random.choice(self.known_label_list)
                    example.label = self.ood_token
                    ood_examples.append(example)


            examples = ind_examples + ood_examples

        if args.ood_type == 'far':
            set_seed(args.seed)

            ind_examples = []
            ood_noise_examples = []

            ori_noise_examples = processor.get_examples(self.open_noise_data_dir, mode)


            known_examples = [e for e in ori_examples if e.label in self.known_label_list]
            num_known_samples = len(known_examples)
            num_open_noise_samples = len(ori_noise_examples)

            num_ind_select = int(num_known_samples * args.ind_noise_ratio / (1 - args.ood_noise_ratio))
            num_ood_select = int(num_known_samples * args.ood_noise_ratio / (1 - args.ood_noise_ratio))


            ind_noise_indices = np.random.choice(num_known_samples, size=num_ind_select, replace=False)
            for idx, example in enumerate(known_examples):
                if idx in ind_noise_indices:
                    possible_labels = [label for label in self.known_label_list if label != example.label]
                    example.noise_label = random.choice(possible_labels)
                else:
                    example.noise_label = example.label
                ind_examples.append(example)


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

            set_seed(args.seed)

            g = torch.Generator()
            g.manual_seed(args.seed)

            sampler = RandomSampler(datatensor, generator=g)

            dataloader = DataLoader(
                datatensor,
                sampler=sampler,
                batch_size=args.train_batch_size,
                drop_last=False,
                num_workers=0,
                generator=g
            )
            # =========================================================

        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(datatensor)
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size=args.eval_batch_size, num_workers=0)

        return dataloader
    def get_loader_(self, examples, args, mode='train'):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(examples, self.label_list_includeood, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_noiseids = torch.tensor([f.label_noiseid for f in features], dtype=torch.long)
        indices = torch.arange(len(features), dtype=torch.long)
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



class InputExample(object):
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
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines



class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")),
                "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")

        labels = np.unique(np.array(test[
                                        'label']))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(

                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(
            example.text_a)


        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        else:

            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(
            tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        label_noiseid=label_map[example.noise_label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          label_noiseid=label_noiseid
                          ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()

def save_dataloader_index(indices, save_path):
    dir_path = os.path.dirname(save_path)
    if dir_path:
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