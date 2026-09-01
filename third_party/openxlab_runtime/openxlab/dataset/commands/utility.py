from functools import wraps
import sys
from typing import Any
from typing import Callable
from typing import TypeVar

from openxlab.dataset.client.client import Client
from openxlab.dataset.constants import endpoint
from openxlab.dataset.exception import OpenDataLabError


_Callable = TypeVar("_Callable", bound=Callable[..., None])


def exception_handler(func: _Callable) -> _Callable:
    """Decorator for CLI functions to catch custom exceptions.

    Arguments:
        func: The CLI function needs to be decorated.

    Returns:
        The CLI function with exception catching procedure.

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            func(*args, **kwargs)
        except OpenDataLabError as err:
            if err.STATUS_CODE == 401:
                print("Error: authentication failure, please login!")
                pass
            elif err.STATUS_CODE == 403:
                print("Unable to access. Please visit the dataset homepage!")
                pass
            elif err.STATUS_CODE == 404:
                print("Data not exists!")
                pass
            elif err.STATUS_CODE == 412:
                print("Access with cdn error!")
                pass
            elif err.STATUS_CODE == 500:
                print("Internal server occurs!")
                pass
            else:
                print("Error occurs!!!")

            sys.exit(1)

    return wrapper  # type: ignore[return-value]


class ContextInfoNoLogin:
    """This class contains command context, the command no need to login"""

    def __init__(self):
        self.url = endpoint

    def get_client(self) -> Client:
        return Client(host=self.url, odl_cookie='')
