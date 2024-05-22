import shlex
import subprocess

from huggingface_hub import HfApi


def hub_upload_check(push_to_hub: str):
    if push_to_hub is not None:
        # Try creating a test repo
        test_repo_name = f"{push_to_hub}-dummy-test"
        try:
            HfApi().create_repo(
                repo_id=test_repo_name,
                repo_type="model",
                private=True
            )
            HfApi().delete_repo(
                repo_id=test_repo_name,
                repo_type="model"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to push test repo {test_repo_name} to HuggingFace Hub. Please check your permissions and network connection. Use `huggingface-cli login` to log in your account.\n\n{str(e)}")


def hub_upload_model_async(push_to_hub: str, push_to_hub_delete_local: bool, save_path: str, epoch: int):
    if push_to_hub is not None:
        safe_repo_name = shlex.quote(f"{push_to_hub}-ep-{epoch}")
        safe_save_path = shlex.quote(save_path)

        command = f"huggingface-cli upload --quiet --repo-type model --private {safe_repo_name} {safe_save_path}"
        if push_to_hub_delete_local:
            command += f" && rm -rf {safe_save_path}"

        subprocess.Popen(command, shell=True)
