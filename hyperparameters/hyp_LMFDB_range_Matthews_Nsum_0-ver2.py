
# Define hyperparameters and system_params

import math
import parse
import primesieve as ps

from file_ops import get_LMFDB_all_file_names


max_prime = 10**3
data_length = len(ps.primes(max_prime))
stride = 2
# data_length and convolution stride directly determine the number of required reducing levels
reducing_layers = int(math.ceil(math.log(data_length)/math.log(stride)))
assert reducing_layers >= 3

# Model hyperparameters
hyperparameters = dict(
    model_name="LMFDB_range_Matthews_Nsum_0_p1000-ver2",
    model_num=None,
    model_type="SimpleFCClassificationModel",
    model=dict(
        nonlin_str="ReLU_inplace",
        hidden_lin_ftrs=128,
        hidden_lin_ftrs_no=4,
        dropout="OPTIMIZE: lambda idx: float(idx) # continuous # (0.0, 0.99)",
    ),
    dataloader=dict(
        data_files=get_LMFDB_all_file_names(),
        dataset="NSumDataset",
        load_reduced_metadata=False,
        max_prime=max_prime,
        skip_first_primes=0,  # change to drop some number of first ap-s - DO NOT USE
        valid_size=0.2,  # relative percentage of data after removing test data
        test_size=None,  # percentage of whole data
        min_max_conductor=[1, 10**8],
        min_max_test_conductor=[10**8+1, 10**9],
        curves_count=None,
        test_curves_count=None,
        purge_repetitive_curves=False,
        sort=None,
        min_max_rank=None,
        only_prime_conductors=False,
        only_not_prime_conductors=False,
        class_identifier="rank",
        min_class_size=50,
        batch_size=2048,
        test_batch_size=2048,  # for test
        test_RAND_SEED=42,
        valid_RAND_SEED=42,
        shuffle_RAND_SEED=42,
        normalize_aps=True,  # divide aps by sqrt(p)
        use_p=False,  # use additional data about normalized p
        use_sqrt_p=False,  # use additional data about normalized sqrt(p)
        use_log_p=False,  # use additional data about normalized log(p)
        use_conductors=True,  # use additional data about normalized log10 of conductors
        # use mask to select variants of Nagao-Mestre sums to use - used only for NSumDataset
        N_masks=[1, 0, 0, 0],
        load_save_cache=True,
    ),
    opt_func_str="Adam",
    loss_func_str="CrossEntropyLoss_weighted",
    epochs=10,
    optimizer=dict(
        lr_max="OPTIMIZE: lambda idx: float(10 ** idx) # continuous # (-6, 0)",
        wd="OPTIMIZE: lambda idx: float(10 ** idx) # continuous # (-10, 0)",
    ),
)


def compute_gyopt_metric(lines, ILLEGAL_val=-1):
    """Computes optimization metric from captured training process console output.
    Implements model and problem specific metric.

    Args:
        lines (list of str): training process console output

    Raises:
        RuntimeError: Unexpected training process console output

    Returns:
        float: computed problem specific metric
    """
    vl_tl_ratio_tolerance = 0.1  # problem specific parameter

    # Parse output lines
    Matthews_corrcoef = None
    vl_tl_ratio = None
    model_not_trained = None
    for line in lines:
        # Parse Matthews_corrcoef
        parsed = parse.parse("{}'Matthews_corrcoef': {coef:g}{}", line)
        if parsed is not None:
            Matthews_corrcoef = parsed.named['coef']

        # Parse vl_tl_ratio
        parsed = parse.parse("{}'vl_tl_ratio': {ratio:g}{}", line)
        if parsed is not None:
            vl_tl_ratio = parsed.named['ratio']

        # Parse model_not_trained
        parsed = parse.parse("{}'model_not_trained': {flag:l}{}", line)
        if parsed is not None:
            if parsed.named['flag'] == "True":
                model_not_trained = True
            elif parsed.named['flag'] == "False":
                model_not_trained = False
            else:
                raise RuntimeError(
                    f"'model_not_trained' has illegal value {parsed.named['flag']}")

    # Compute and return metric
    if model_not_trained is None:
        raise RuntimeError(
            "Property 'model_not_trained' is not present in results.")
    elif model_not_trained:
        return ILLEGAL_val
    else:
        if Matthews_corrcoef is None:
            raise RuntimeError("No Matthews correlation coefficient computed.")
        elif vl_tl_ratio is None:
            raise RuntimeError("No vl_tl_ratio computed.")
        else:
            metric = Matthews_corrcoef
            return metric
