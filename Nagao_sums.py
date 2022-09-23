
# Computation of several variants of Nagao-Mestre sums

import logging
import torch
import math
from tqdm import tqdm
import primesieve as ps

from file_ops import get_beg_end_classes_list


def compute_Nagao_sums(aps, max_prime, bad_primes, MAX_MEM_SIZE=10**9):
    """Compute several variants of Nagao-Mestre sums.

    Args:
        aps (torch.Tensor): array in each row having ap-s for a single curve
        max_prime (int): upper bound for used primes
        bad_primes (torch.Tensor): boolean array of the same shape as aps, having 1 where prime divides the conductor, 0 otherwise

    Returns:
        tuple : several variants of computed sums
    """
    if aps.size(0) * aps.size(-1) > MAX_MEM_SIZE:
        length = int(math.floor(MAX_MEM_SIZE / aps.size(-1)))
        logging.info("Computing Nagao sums in smaller batches to conserve RAM...")
        out_t=[]
        for beg_idx, end_idx in tqdm(get_beg_end_classes_list(0, aps.size(0), length)):
            logging.disable(logging.INFO)
            out_t.append(compute_Nagao_sums(aps[beg_idx:end_idx], max_prime, bad_primes[beg_idx:end_idx]))
            logging.disable(logging.NOTSET)
        out = torch.cat(out_t)
        return out   
    else:
        # strange hack, possibly because of bug in pytorch
        cuda_device = aps.get_device() if aps.get_device() != -1 else 'cpu'

        # compute list of primes used
        num_of_primes = ps.count_primes(max_prime)
        if aps.shape[-1] < num_of_primes:
            raise ValueError(
                f"Number of primes smaller or equal to {max_prime} is greater than the size {aps.shape[-1]} of supplied ap-s")
        primes = list(ps.primes(max_prime))

        # compute bad primes mask
    #    logging.info("Computing divisions of conductors by primes...")
    #    bad_primes = torch.BoolTensor(
    #        [[cond % p == 0 for p in primes] for cond in tqdm([int(c) for c in conductors])], device=cuda_device)

        logging.info("Computing mask for bad primes...")
        # mask iz 0 for bad prime and 1 for good prime
        bad_primes_mask = torch.where(bad_primes.to(torch.bool), torch.zeros(bad_primes.shape, dtype=torch.float32, device=cuda_device),
                                    torch.ones(bad_primes.shape, dtype=torch.float32, device=cuda_device))

        # all primes should be represented in float32 precision exactly
        assert max_prime <= 10**6
        # convert primes to torch
        primes = torch.FloatTensor(primes, device=cuda_device)

        logging.info("Computing Nagao sums...")
        return torch.stack((torch.sum(aps *
                                    (torch.log(primes) / primes).unsqueeze(0).expand(aps.size(0), -1) *
                                    bad_primes_mask, 1, dtype=torch.float64) / torch.log(primes[-1]),
                            torch.sum((-aps + 2) /
                                    (primes.unsqueeze(0).expand(aps.size(0), -1) + 1 - aps) *
                                    torch.log(primes).unsqueeze(0).expand(aps.size(0), -1) *
                                    bad_primes_mask, 1, dtype=torch.float64),
                            torch.sum(-aps *
                                    torch.log(primes).unsqueeze(0).expand(aps.size(0), -1) *
                                    bad_primes_mask, 1, dtype=torch.float64) / primes[-1],
                            torch.sum(torch.log((primes.unsqueeze(0).expand(aps.size(0), -1) + 1 - aps) /
                                                primes.unsqueeze(0).expand(aps.size(0), -1)) *
                                    bad_primes_mask +
                                    torch.log(((primes.unsqueeze(0).expand(aps.size(0), -1) - 1) * 1.5) /  # arbitrary coefficient in [1,2] ?
                                                primes.unsqueeze(0).expand(aps.size(0), -1)) *
                                    (1 - bad_primes_mask),
                                    1, dtype=torch.float64),
                            ), dim=1).to(torch.float32)
