from utils import parse_args, load_config, Applier_Context


if __name__ == '__main__':
    cfg = parse_args()
    applier = Applier_Context(cfg)

    applier.get_context(
        threshold=5, 
        max_length=128, 
        max_per_token=10, 
        lines=4
    )
    
