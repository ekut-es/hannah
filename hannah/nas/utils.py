import numpy as np


def is_pareto(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] > c, axis=1
                )  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )  # Remove dominated points
            is_efficient[i] = 1

    return is_efficient
