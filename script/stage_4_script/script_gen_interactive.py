import os, sys, torch

# make project root importable
HERE      = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ_ROOT)

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_TextGen_RNN      import Method_TextGen_RNN

if __name__ == '__main__':
    # ---- set your six start words here ----
    prefix_words = ["what", "did", "the", "bartender", "say", "to"]
    # ----------------------------------------

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1) load vocab & embeddings
    data_obj = Dataset_Loader_Generation(
        'text_generation', 'jokes', emb_dim=100
    )
    data_obj.dataset_source_folder_path = os.path.join(
        PROJ_ROOT, 'data', 'stage_4_data', 'text_generation'
    )
    data_obj.dataset_source_file_name = 'data'
    data_obj.load()  # builds stoi/itos/embedding_matrix

    # 2) instantiate model & load checkpoint
    arch = 'lstm'   # or 'rnn', 'gru' – choose the variant you trained
    vocab_size = len(data_obj.itos)
    model = Method_TextGen_RNN(
        'TextGenRNN','joke gen',
        vocab_size=vocab_size,
        emb_dim=100,
        hidden_size=128,
        num_layers=1,
        rnn_arch=arch,
        max_epoch=0, learning_rate=0,
        batch_size=1,
        pad_idx=data_obj.pad_idx,
        sos_idx=data_obj.sos_idx,
        eos_idx=data_obj.eos_idx
    )
    # load trained weights
    ckpt = os.path.join(
        PROJ_ROOT, 'result','stage_4_result','gen',
        f"{arch}_model.pt"
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    # inject GloVe embeddings
    model.embedding.weight.data.copy_(data_obj.embedding_matrix.to(device))

    # 3) generate continuation
    prefix_ids = [data_obj.stoi.get(w, data_obj.unk_idx) for w in prefix_words]
    gen_ids = model.generate(prefix_ids, max_gen_len=200)
    gen_toks = [data_obj.itos[i] for i in gen_ids]
    generated = " ".join(prefix_words + gen_toks)

    # 4) print to console
    print(">> Prefix :", " ".join(prefix_words))
    print(">> Generated:", generated)

    # 5) append to oodInfer.txt
    out_dir = os.path.join(PROJ_ROOT, 'result','stage_4_result','gen')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'oodInfer.txt')
    with open(out_path, 'a', encoding='utf8') as f:
        f.write(generated + "\n")

    print(f"[Appended generation to] {out_path}")
