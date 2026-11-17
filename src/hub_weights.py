"""
Share trained weights without committing them to Git (assignment rule R7).

Upload your trained models/ folder to a Hugging Face Hub model repo:
    huggingface-cli login
    python -m src.hub_weights upload --repo <your-hf-username>/support-ticket-router

Download them on any machine (then `python app.py` works without training):
    python -m src.hub_weights download --repo <your-hf-username>/support-ticket-router
"""
from __future__ import annotations

import argparse

from huggingface_hub import HfApi, snapshot_download

from src import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["upload", "download"])
    ap.add_argument("--repo", required=True, help="e.g. username/support-ticket-router")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if args.action == "upload":
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(C.MODELS_DIR), repo_id=args.repo, repo_type="model",
                          commit_message="Upload trained router models")
        print(f"Uploaded {C.MODELS_DIR} -> https://huggingface.co/{args.repo}")
    else:
        snapshot_download(repo_id=args.repo, repo_type="model", local_dir=str(C.MODELS_DIR))
        print(f"Downloaded https://huggingface.co/{args.repo} -> {C.MODELS_DIR}")


if __name__ == "__main__":
    main()
