from utils import parse_args, load_config, Applier_ChosenActivation


if __name__ == '__main__':
    cfg = parse_args()
    applier = Applier_ChosenActivation(cfg)

    applier.get_context(
        threshold=0, 
        max_length=0, 
        max_per_token=0, 
        lines=0
    )
    
