import argparse
from ugafl import UGAFL


def main(args):
    simulator = UGAFL(3)

    for t in range(args.num_rounds):
        simulator.run_single_round(t)
        # visualize the server's model parmaeters at the end of each round
        simulator.visualize_model(save=True, filename=f"model_round_{t}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rounds', type=int, default=20,
                        help='number of rounds to run')
    args = parser.parse_args()

    main(args)
