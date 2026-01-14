import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed.')  # seed=0

    parser.add_argument("--met_in_channels",
                        type=int,
                        default=265,
                        help="metabolites input feature dimensions. Default is 265.")

    parser.add_argument("--met_hidden",
                        type=int,
                        default=64,
                        help="metabolites hidden feature dimensions. Default is 128.")

    parser.add_argument("--met_out_channels",
                        type=int,
                        default=64,
                        help="metabolites output feature dimensions. Default is 64.")

    parser.add_argument("--dis_in_channels",
                        type=int,
                        default=2315,
                        help="diseases input feature dimensions. Default is 2315.")

    parser.add_argument("--dis_hidden",
                        type=int,
                        default=128,
                        help="diseases hidden feature dimensions. Default is 128.")

    parser.add_argument("--dis_out_channels",
                        type=int,
                        default=64,
                        help="diseases output feature dimensions. Default is 64.")

    parser.add_argument('--lr',
                        type=float,
                        default=0.005,  # lr=0.01
                        help='Initial learning rate.')

    parser.add_argument('--epochs',
                        type=int,
                        default=2000,
                        help='Number of epochs to train.')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.3,  # dropout=0.3 为最优
                        help='Dropout rate.')

    return parser.parse_args()