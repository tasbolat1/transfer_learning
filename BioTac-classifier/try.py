import argparse

def main():

    # create the parser
    parser = argparse.ArgumentParser(description="Train a BioTac classifier")

    # add the arguments
    parser.add_argument("--cutidx", type=int, default=None, help="length for cutting the time window of sliding data")
    args = parser.parse_args()

    print(args.cutidx)

if __name__ == "__main__":
    main()