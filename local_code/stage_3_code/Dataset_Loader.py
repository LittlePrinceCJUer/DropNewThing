from local_code.base_class.dataset import dataset
import pickle, torch, os

class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data for', self.dataset_name, '...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        def proc(inst_list):
            Xs, ys = [], []
            for inst in inst_list:
                img = inst['image']
                name = self.dataset_name.lower()
                if name == 'orl':
                    # grayscale repeated in each channel → pick channel 0
                    img = img[:, :, 0]
                    t = torch.from_numpy(img).unsqueeze(0).float()  # 1×112×92
                elif name == 'mnist':
                    # already 28×28
                    t = torch.from_numpy(img).unsqueeze(0).float()  # 1×28×28
                else:
                    # CIFAR: 32×32×3 → 3×32×32
                    t = torch.from_numpy(img).permute(2, 0, 1).float()
                t = t / 255.0   # normalize
                Xs.append(t)
                ys.append(inst['label'])
            return torch.stack(Xs), torch.tensor(ys, dtype=torch.long)

        X_tr, y_tr = proc(data['train'])
        X_te, y_te = proc(data['test'])
        return {'train': {'X': X_tr, 'y': y_tr},
                'test' : {'X': X_te, 'y': y_te}}