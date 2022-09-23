# Define gyopt metric

import parse


def compute_gyopt_metric(lines, ILLEGAL_val=-1):
    """Computes optimization metric from captured training process console output.
    Implements model and problem specific metric.
    Uses Matthews_corrcoef from binary confusion matrix.

    Args:
        lines (list of str): training process console output

    Raises:
        RuntimeError: Unexpected training process console output

    Returns:
        float: computed problem specific metric
    """

    # Parse output lines
    Matthews_corrcoef_binary = None
    vl_tl_ratio = None
    model_not_trained = None
    for line in lines:
        # Parse Matthews_corrcoef_binary
        parsed = parse.parse("{}'Matthews_corrcoef_binary_classification': {coef:g}{}", line)
        if parsed is not None:
            Matthews_corrcoef_binary = parsed.named["coef"]

        # Parse vl_tl_ratio
        parsed = parse.parse("{}'vl_tl_ratio': {ratio:g}{}", line)
        if parsed is not None:
            vl_tl_ratio = parsed.named["ratio"]

        # Parse model_not_trained
        parsed = parse.parse("{}'model_not_trained': {flag:l}{}", line)
        if parsed is not None:
            if parsed.named["flag"] == "True":
                model_not_trained = True
            elif parsed.named["flag"] == "False":
                model_not_trained = False
            else:
                raise RuntimeError(
                    f"'model_not_trained' has illegal value {parsed.named['flag']}"
                )

    # Compute and return metric
    if model_not_trained is None:
        raise RuntimeError("Property 'model_not_trained' is not present in results.")
    elif model_not_trained:
        return ILLEGAL_val
    else:
        if Matthews_corrcoef_binary is None:
            raise RuntimeError("No Matthews correlation coefficient computed.")
        elif vl_tl_ratio is None:
            raise RuntimeError("No vl_tl_ratio computed.")
        else:
            metric = Matthews_corrcoef_binary
            return metric