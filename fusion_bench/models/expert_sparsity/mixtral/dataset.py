import torch


class CacheDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.alphas = []  # logits
        self.Xs = []  # input hidden states
        self.Zs = []  # output hidden states
        self.prepared = False

    def __len__(self):
        if not self.prepared:
            self.prepare_for_loader()
        return len(self.alphas)

    def __getitem__(self, index):
        if not self.prepared:
            self.prepare_for_loader()
        if isinstance(index, list):
            return [(self.alphas[idx], self.Xs[idx], self.Zs[idx]) for idx in index]
        elif isinstance(index, int):
            return self.alphas[index], self.Xs[index], self.Zs[index]

    def append(self, alpha=None, X=None, Z=None):
        if alpha is not None:
            self.alphas.append(alpha.detach().to("cpu", non_blocking=True))
        if X is not None:
            self.Xs.append(X.detach().to("cpu", non_blocking=True))
        if Z is not None:
            self.Zs.append(Z.detach().to("cpu", non_blocking=True))
        self.prepared = False

    def prepare_for_loader(self):
        if self.prepared:
            return
        self.prepared = True
        self.alphas = torch.concat(self.alphas)
        self.Xs = torch.concat(self.Xs)
        self.Zs = torch.concat(self.Zs)
        assert len(self.Xs) == len(self.Zs)
