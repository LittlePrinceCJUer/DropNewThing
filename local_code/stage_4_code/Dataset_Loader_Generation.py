from local_code.base_class.dataset import dataset
import os, torch
from torchtext.vocab import GloVe

class Dataset_Loader_Generation(dataset):
    """
    Reads:
      data/stage_4_data/text_generation/data
    Builds:
      - self.stoi, self.itos, self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx
      - self.embedding_matrix (vocab_size × emb_dim)
      - self.raw_jokes: list of token lists

    Returns:
      {'train': {'X': LongTensor[N, L], 'y': LongTensor[N, L]}}
      where L = max_original_length + 2  (<sos> + tokens + <eos>)
    """

    def __init__(self, dName=None, dDescription=None,
                 emb_dim=100):
        super().__init__(dName, dDescription)
        self.emb_dim = emb_dim
        # load GloVe for embedding init
        self.glove = GloVe(name='6B', dim=emb_dim)

    def load(self):
        # 1) read raw lines
        path = os.path.join(self.dataset_source_folder_path,
                            self.dataset_source_file_name)
        texts = []
        with open(path, encoding='utf8', errors='ignore') as f:
            next(f)  # skip header
            for line in f:
                _id, rest = line.strip().split(',', 1)
                joke = rest.strip().strip('"')
                texts.append(joke)

        # 2) whitespace tokenize (preserve punctuation)
        tokenized = [t.split() for t in texts]
        self.raw_jokes = tokenized

        # 3) build vocab
        specials = ['<pad>','<unk>','<sos>','<eos>']
        uniq = {tok for seq in tokenized for tok in seq}
        itos = specials + sorted(uniq)
        stoi = {tok: i for i, tok in enumerate(itos)}

        self.stoi, self.itos = stoi, itos
        self.pad_idx = stoi['<pad>']
        self.unk_idx = stoi['<unk>']
        self.sos_idx = stoi['<sos>']
        self.eos_idx = stoi['<eos>']

        # 4) build embedding matrix
        V = len(itos)
        emb_mat = torch.zeros(V, self.emb_dim)
        for tok, idx in stoi.items():
            if tok in self.glove.stoi:
                emb_mat[idx] = self.glove.vectors[self.glove.stoi[tok]]
        self.embedding_matrix = emb_mat

        # 5) prepare sequences
        # determine max original length M (no <sos>/<eos>)
        M = max(len(seq) for seq in tokenized)
        L = M + 2  # <sos> + seq + <eos>

        seqs_in, seqs_out = [], []
        for seq in tokenized:
            # input: <sos> + tokens + <eos>
            inp = [self.sos_idx] + [stoi.get(tok, self.unk_idx) for tok in seq] + [self.eos_idx]
            # target: tokens + <eos> + pad (implicitly)
            tgt = [stoi.get(tok, self.unk_idx) for tok in seq] + [self.eos_idx]

            # pad each to length L
            pad_inp = L - len(inp)
            pad_tgt = L - len(tgt)
            inp = inp + [self.pad_idx] * pad_inp
            tgt = tgt + [self.pad_idx] * pad_tgt

            seqs_in.append(torch.tensor(inp, dtype=torch.long))
            seqs_out.append(torch.tensor(tgt, dtype=torch.long))

        X = torch.stack(seqs_in)  # [N, L]
        y = torch.stack(seqs_out) # [N, L]

        return {'train': {'X': X, 'y': y}}
