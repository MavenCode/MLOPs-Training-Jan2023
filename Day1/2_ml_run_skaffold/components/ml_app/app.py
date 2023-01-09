import sys
import argparse

def main(user: str) -> None:
    print(f"Hello, {user}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hello world MLOps"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="the person to greet",
        required=True,
    )
    args = parser.parse_args(sys.argv[1:])

    main(args.user)