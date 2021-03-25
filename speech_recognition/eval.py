import hydra


@hydra.main(config_name="eval", config_path="conf")
def main():
    "Neural network evaluation module"
