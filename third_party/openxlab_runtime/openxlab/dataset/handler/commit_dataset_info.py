"""
commit message of change of dataset repository
"""
from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.xlab.handler.user_token import trigger_update_check


def commit(dataset_repo: str, commit_message: str):
    """
    Commit local changes.

    Example:
        openxlab.dataset.commit(dataset_repo="username/dataset_repo_name",commit_message="init")

    Parameters:
        @dataset_repo String The address of the dataset repository to commit.
        @commit_message String The detail message to commit of the repository.
    """
    # update check
    trigger_update_check()

    ctx = ContextInfoNoLogin()
    client = ctx.get_client()

    req_data_list = [f"{dataset_repo}", {"msg": f"{commit_message}"}]
    client.get_api().commit_dataset(req=req_data_list)
