import argparse


def extract_rate_and_size(task_description: str, unit_size: int) -> tuple[float, int]:
    parser = argparse.ArgumentParser(description=task_description)

    parser.add_argument("-s", "--unit_size",
                        type=int,
                        default=unit_size,
                        help="The size of each block in bytes.")

    parser.add_argument("-r", "--rate",
                        type=float,
                        required=True,
                        help="The number of packets per second to send.")

    args = parser.parse_args()
    return args.rate, args.unit_size
