from huggingface_hub import snapshot_download
from dotenv import load_dotenv
load_dotenv(override=True)

def model_download(model_name: str):
    """
    從 HuggingFace 下載模型

    Args:
        model_name (str): 模型名稱，如：meta-llama/Meta-Llama-3-8B
    """

    snapshot_download(
        repo_id=model_name
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name, Ex: meta-llama/Meta-Llama-3-8B")
    args = parser.parse_args()

    if args.model is None:
        raise ValueError("model name is required")
    
    model_download(args.model)