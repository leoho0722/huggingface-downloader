from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from dotenv import load_dotenv
load_dotenv(override=True)


def model_download(args):
    """
    從 HuggingFace 下載模型

    Args:
        model_name (str): 模型名稱，如：meta-llama/Meta-Llama-3-8B
    """

    snapshot_download(
        repo_id=args.model,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model name, Ex: meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument(
        "--cache_dir",
        default=HF_HUB_CACHE,
        help="Model cache directory"
    )
    args = parser.parse_args()

    if args.model is None:
        raise ValueError("model name is required")

    model_download(args)
