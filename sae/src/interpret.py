from utils import parse_args, load_config, Interpreter


if __name__ == '__main__':
    cfg = parse_args()
    interp = Interpreter(cfg)
    interp.run(sample_latents=300)