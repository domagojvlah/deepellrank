
# Datasets definitions

import logging
import torch

from torch.utils.data import TensorDataset


class EllipticCurveDataset(TensorDataset):
    def __init__(self, aps, label, primes, conductors, log10conductors, curves,
                 normalize_aps=True,
                 use_p=False,
                 use_sqrt_p=False,
                 use_log_p=False,
                 use_conductors=True,
                 ):
        assert len(aps.shape) == 2
        assert len(label.shape) == 1
        assert len(primes.shape) == 1
        assert aps.size(0) == label.size(0)
        assert aps.size(1) == primes.size(0)
        assert aps.size(0) == len(conductors)
        assert aps.size(0) == len(curves)

        self.normalize_aps = normalize_aps
        self.use_p = use_p
        self.use_sqrt_p = use_sqrt_p
        self.use_log_p = use_log_p
        self.use_conductors = use_conductors
        self.primes = primes
        self.conductors = conductors
        self.log10conductors = log10conductors
        self.max_log10conductors = max(self.log10conductors)
        self.curves = curves
        self.data_channels = 1 + \
            int(use_p) + int(use_sqrt_p) + int(use_log_p) + int(use_conductors)
        super().__init__(aps, label)

    def override_max_log10conductors(self, max_log10conductors):
        logging.info(
            f"  Overriding log10 of maximum conductor computed from from dataset {self.max_log10conductors} with {max_log10conductors}")
        self.max_log10conductors = max_log10conductors

    def get_curves(self, idxs):
        if isinstance(idxs, int):
            return self.curves[idxs]
        elif isinstance(idxs, list):
            assert set(type(idx) for idx in idxs) == {int}
            return [self.curves[idx] for idx in idxs]
        else:
            raise TypeError(f"idx is not of type int or list")

    def __getitem__(self, idx):
        aps, label = super().__getitem__(idx)

        # Make several transformations and concatenations to data tensor including: normalization, primes, sqrt of primes, log of primes

        if self.normalize_aps:
            data = aps.float() / torch.sqrt(self.primes)
        else:
            data = aps.float()
        data = data.unsqueeze(0)

        if self.use_p:
            data_2 = self.primes.clone().detach()
            data_2 = data_2 / data_2[-1]
            data_2 = data_2.unsqueeze(0)
            data = torch.cat((data, data_2), 0)

        if self.use_sqrt_p:
            data_2 = torch.sqrt(self.primes.clone().detach())
            data_2 = data_2 / data_2[-1]
            data_2 = data_2.unsqueeze(0)
            data = torch.cat((data, data_2), 0)

        if self.use_log_p:
            data_2 = torch.log(self.primes.clone().detach())
            data_2 = data_2 / data_2[-1]
            data_2 = data_2.unsqueeze(0)
            data = torch.cat((data, data_2), 0)

        if self.use_conductors:
            data_2 = torch.tensor(self.log10conductors[idx]/self.max_log10conductors,
                                  dtype=torch.float32,
                                  device=label.device)
            data_2 = data_2.unsqueeze(0).expand(self.primes.size(0))
            data_2 = data_2.unsqueeze(0)
            data = torch.cat((data, data_2), 0)

        return data, label


class NSumDataset(TensorDataset):
    def __init__(self, Nsums, label, conductors, log10conductors, curves,
                 N_masks=[1, 1, 1, 1],
                 use_conductors=True,
                 ):
        assert Nsums.size(-1) == len(N_masks)
        self.conductors = conductors
        self.log10conductors = log10conductors
        self.max_log10conductors = max(self.log10conductors)
        self.curves = curves
        self.data_channels = sum(N_masks) + int(use_conductors)
        self.use_conductors = use_conductors
        self.device = label.device

        N_masks_idxs = torch.tensor([idx for idx, mask in enumerate(
            N_masks) if mask == 1], dtype=torch.int, device=Nsums.device)
        logging.info(f"  Indices of Nagao sums used: {N_masks_idxs.tolist()}")
        self.Nsums_selected = torch.index_select(Nsums, 1, N_masks_idxs)

        self.calculate_Nsums()
        self.label = label

    def calculate_Nsums(self):
        if self.use_conductors:
            logging.info(
                f"  Using conductors in dataset. Max log10 of conductors is {self.max_log10conductors}. Normalizing conductors...")
            cond_t = torch.tensor(self.log10conductors,
                                  dtype=torch.float32, device=self.device).unsqueeze(-1)
            cond_t = cond_t / self.max_log10conductors
            self.Nsums = torch.cat((self.Nsums_selected, cond_t), dim=-1)
        else:
            logging.info("  Not using conductors in dataset.")
            self.Nsums = self.Nsums_selected

    def override_max_log10conductors(self, max_log10conductors):
        logging.info(
            f"  Overriding log10 of maximum conductor computed from from dataset {self.max_log10conductors} with {max_log10conductors}")
        self.max_log10conductors = max_log10conductors
        self.calculate_Nsums()

    def get_curves(self, idxs):
        if isinstance(idxs, int):
            return self.curves[idxs]
        elif isinstance(idxs, list):
            assert set(type(idx) for idx in idxs) == {int}
            return [self.curves[idx] for idx in idxs]
        else:
            raise TypeError(f"idx is not of type int or list")

    def __getitem__(self, idx):
        Nsums, label = self.Nsums[idx], self.label[idx]

        # if self.use_conductors:
        #     cond_t = torch.zeros(1, dtype=torch.float32, device=label.device)
        #     cond_t[0] = self.log10conductors[idx]/self.max_log10conductors
        #     return torch.cat((Nsums, cond_t)), label
        # else:
        return Nsums, label
