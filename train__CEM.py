


def main():
    args = parse_args()

    # Print arguments
    print("\n\n\t\t Arguments:")
    for arg in vars(args):
        print(f"\t\t\t {arg}: {getattr(args, arg)}")

    # Set device (handle CUDA, MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")

    # Create output folder based on arguments
    output_folder = create_output_folder(args)

    args.output_folder = output_folder

    # Save all arguments to a JSON file
    args_path = os.path.join(output_folder, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_path}")

    args.device = device
    print("\n\n\t\t Starting training on device: ", device)
    print("\n\n\t\t Output folder: ", output_folder)