"""
delete dataset repository for user
"""
from rich import print as rprint

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.xlab.handler.user_token import trigger_update_check


def remove_repo(dataset_repo: str):
    """
    Remove a dataset repository.

    Example:
        openxlab.dataset.remove_repo(dataset_repo="username/dataset_repo_name")

    Parameters
        @dataset_repo String The address of dataset repository.
    """
    # update check
    trigger_update_check()

    ctx = ContextInfoNoLogin()
    client = ctx.get_client()
    parsed_ds_name = dataset_repo.replace("/", ",")
    rprint(f"Removing dataset repository [red]{dataset_repo}[/red]...")
    client.get_api().delete_repo(parsed_ds_name)
    rprint(f"Dataset repository [blue]{dataset_repo}[/blue] removed successfully.")
