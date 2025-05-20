from local_code.base_class.dataset import dataset
import os, torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
import string

class Dataset_Loader(dataset):
    """
    Loads text files under:
      data/stage_4_data/text_classification/{train,test}/{neg,pos}/*.txt
    Tokenizes, removes basic stopwords/punctuation, maps to GloVe embeddings,
    pads or truncates to fixed length.
    """
    def __init__(self, dName=None, dDescription=None, max_len=100, emb_dim=100):
        super().__init__(dName, dDescription)
        self.max_len = max_len
        self.emb_dim = emb_dim
        # basic English tokenizer
        self.tokenizer = get_tokenizer('basic_english')
        # minimal stopword list
        self.stopwords = set(["a","an","the","in","on","at","and","or","is","are","to","of","for","with","that","this","it"])
        # load GloVe
        self.glove = GloVe(name='6B', dim=self.emb_dim)

    def _process_split(self, split_dir):
        Xs, ys = [], []
        for label, cls in enumerate(['neg','pos']):
            cls_dir = os.path.join(split_dir, cls)
            for fn in os.listdir(cls_dir):
                if not fn.endswith('.txt'): continue
                path = os.path.join(cls_dir, fn)
                text = open(path, 'r', encoding='utf8').read()
                # tokenize, lowercase, remove punctuation & stopwords
                tokens = [tok for tok in self.tokenizer(text)
                          if tok not in self.stopwords and all(ch not in string.punctuation for ch in tok)]
                # map to embeddings
                if len(tokens)==0:
                    vecs = torch.zeros((0, self.emb_dim))
                else:
                    vecs = self.glove.get_vecs_by_tokens(tokens, lower_case_backup=True)
                # pad/truncate
                if vecs.size(0) >= self.max_len:
                    vecs = vecs[:self.max_len]
                else:
                    pad = torch.zeros((self.max_len-vecs.size(0), self.emb_dim))
                    vecs = torch.cat([vecs, pad], dim=0)
                Xs.append(vecs)         # [max_len, emb_dim]
                ys.append(label)        # 0=neg,1=pos
        X = torch.stack(Xs)          # [N, max_len, emb_dim]
        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    def load(self):
        print(f"loading data for {self.dataset_name}...")
        root = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        )
        X_tr, y_tr = self._process_split(os.path.join(root, 'train'))
        X_te, y_te = self._process_split(os.path.join(root, 'test'))
        return {'train': {'X': X_tr, 'y': y_tr},
                'test' : {'X': X_te, 'y': y_te}}