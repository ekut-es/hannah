from hannah.models.embedded_vision_net.models import embedded_vision_net
from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.sampler.aging_evolution import AgingEvolutionSampler
from omegaconf import OmegaConf
from hannah.nas.functional_operators.op import Tensor
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
import numpy as np
import pandas as pd
from tqdm import tqdm


def sample_candidates(num_total, num_candidates, constraint_model, sampler, space):
    print(f"Sample candidates {num_candidates}|{num_total}")
    cands = []
    pbar = tqdm(total=num_total)
    while len(cands) < num_total:
        parameters, keys, parent_index = sampler.next_parameters()
        p_df = pd.DataFrame([parameters])
        # comp = "random"
        if constraint_model:
            while True:
                try:
                    constrainer.solve(space, parameters)
                    parameters = constrainer.get_constrained_params(parameters)
                    cp_df = pd.DataFrame([parameters])
                    # comp = cp_df.compare(p_df)
                    break
                except Exception as e:
                    print("Error occurred while solving constraints: ", str(e))
        # else:
        #     parameters, keys, parent_index = sampler.next_parameters()
        #     comp = "random"

        ve = np.random.rand()
        cands.append({"params": parameters, "val_error": ve, "changes": keys, "parent_index": parent_index})
        pbar.update()
    cands.sort(key=lambda x: x["val_error"])
    candidates = cands[:num_candidates]
    pbar.close()
    return candidates


if __name__ == '__main__':
    input = Tensor("input", shape=(1, 3, 32, 32), axis=("N", "C", "H", "W"))
    cons = OmegaConf.create([{"name": "weights",  "upper": 500000}])
    space = embedded_vision_net("evn", input=input, num_classes=10, constraints=cons)
    cfg = OmegaConf.create({"nas": {"bounds": {"val_error": 0.2}}})
    sampler = AgingEvolutionSampler(parent_config=cfg, search_space=space, parametrization=space.parametrization(), population_size=20, sample_size=5)
    constrainer = RandomWalkConstraintSolver()

    budget = 100
    num_total = 20
    num_candidates = 10
    candidates = sample_candidates(num_total, num_total, constraint_model=None, sampler=sampler, space=space)
    pbar = tqdm(total=budget)
    while len(sampler.history) < budget:
        if len(candidates) == 0:
            candidates = sample_candidates(num_total, num_candidates, constraint_model=constrainer, sampler=sampler, space=space)
        else:
            s = candidates.pop(0)
            set_parametrization(s["params"], space.parametrization())
            total_weights = space.weights.evaluate()
            sampler.tell_result(s["params"], metrics={"val_error": s["val_error"], "total_weights": total_weights, "parent_index": s["parent_index"], "changes": s["changes"]})
        pbar.update(len(candidates))

    pbar.close()
