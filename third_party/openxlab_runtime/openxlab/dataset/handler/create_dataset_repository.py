"""
create new dataset repository
"""

from rich import print as rprint

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.xlab.handler.user_token import trigger_update_check


def create_repo(repo_name: str, private: bool = False):
    """
    Create a dataset repository.

    Example:
        openxlab.dataset.create_repo(repo_name="dataset_repo_name")

    Parameters:
        @repo_name String The name of dataset repository.
    """
    try:
        # update check
        trigger_update_check()

        ctx = ContextInfoNoLogin()
        client = ctx.get_client()

        permission = 'private' if private else 'public'
        req_data_dict = {
            "name": f"{repo_name}",
            "displayname": f"{repo_name}",
            "state": permission,
        }

        resp_data_dict = client.get_api().create_dataset(req=req_data_dict)

        rprint(
            f"Dataset named: [blue]{resp_data_dict['name']}[/blue] created successfully. Dataset is {permission}."
        )
    except Exception as e:
        print(e)
