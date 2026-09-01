"""
update dataset repository
"""
from rich import print as rprint

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.xlab.handler.user_token import trigger_update_check


def visibility(dataset_repo: str, private: bool):
    """
    set dataset visibility.

    Example:
        openxlab.dataset.visibility(
            dataset_repo="username/dataset_repo_name",
            private=True
        )

    Parameters:
        @dataset_repo String The address of dataset repository.
        @private String The visibility permission of repository.
    """
    # update check
    trigger_update_check()

    ctx = ContextInfoNoLogin()
    client = ctx.get_client()
    permission = 'private' if private else 'public'

    # parse dataset_name
    parsed_ds_name = dataset_repo.replace("/", ",")
    # check dataset repository validation to set status as public
    if private:
        client.get_api().check_public_validation(parsed_ds_name)
    client.get_api().set_repo_permission(parsed_ds_name, private)

    rprint(f"Visibility: [blue]{dataset_repo}[/blue] now is {permission}.")
