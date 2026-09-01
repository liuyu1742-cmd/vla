import base64
from datetime import datetime
from datetime import timedelta
import hmac
import json
import os
import subprocess

from filelock import FileLock

from openxlab.config import version as config_version
from openxlab.utils.time_util import get_current_formatted_time
from openxlab.utils.time_util import get_current_time
from openxlab.utils.time_util import get_datetime_from_formatted_str
from openxlab.xlab.clients.auth_client import AuthClient
from openxlab.xlab.clients.version_client import VersionClient
from openxlab.xlab.handler.user_config import get_config
from openxlab.xlab.handler.user_config import get_config_dir
from openxlab.xlab.handler.user_config import get_file_content
from openxlab.xlab.handler.user_config import get_token_path
from openxlab.xlab.handler.user_config import get_version_path


AUTH_CLIENT = AuthClient("https://openapi.openxlab.org.cn/api/v1/sso-be/api/v1/open/")
VERSION_CLIENT = VersionClient("https://openxlab.org.cn/gw/openxlab-xlab/")
# AUTH_CLIENT = AuthClient("https://staging.openxlab.org.cn/api/v1/sso-be/api/v1/open/")
# VERSION_CLIENT = VersionClient("https://staging.openxlab.org.cn/gw/openxlab-xlab/")


def calculate_d(sk, nonce, algorithm):
    if len(sk) == 0 or len(nonce) == 0 or len(algorithm) == 0:
        raise ValueError("sk, nonce and algorith must not be empty")
    try:
        hmac_key = bytearray(sk.strip().encode("utf-8"))
        hmac_obj = hmac.new(
            hmac_key,
            nonce.encode("utf-8"),
            algorithm[4:] if algorithm.startswith("Hmac") else algorithm,
        )
        raw_hmac = hmac_obj.digest()
        return base64.b64encode(raw_hmac).decode("utf-8")
    except KeyError as e:
        raise ValueError("Unsupported hash type: %s" % algorithm) from e
    except Exception as e:
        raise ValueError("Error signing nonce: %s" % str(e)) from e


def get_jwt(ak=None, sk=None, auth=False):
    return get_token(ak, sk, auth=auth).jwt


def get_token(ak=None, sk=None, auth=False):
    local_user_token = get_token_from_local()
    if local_user_token is None:
        return get_token_via_api(ak, sk, auth=auth)
    user_token_expiration_datetime = get_datetime_from_formatted_str(local_user_token.expiration)
    now = get_current_time()
    # add 30s buffer
    buffer_timedelta = timedelta(seconds=30)
    expiration_with_buffer = user_token_expiration_datetime - buffer_timedelta
    if expiration_with_buffer <= now:
        return refresh_token(ak, sk, auth=auth)
    return local_user_token


def get_token_from_local():
    if not os.path.exists(get_token_path()):
        return None
    token_json = get_file_content(get_token_path())
    token_dict = json.loads(token_json)
    return UserToken(**token_dict)


def get_token_via_api(ak=None, sk=None, auth=False):
    user_config = get_config(ak, sk, auth=auth)
    if user_config is None:
        raise ValueError(
            "Local config must not be empty before get token via api. "
            "Please use the 'openxlab config' command to set the config"
        )
    nonce, algorithm = AUTH_CLIENT.auth(user_config.ak)
    d = calculate_d(user_config.sk, nonce, algorithm)
    jwt_dict = AUTH_CLIENT.get_jwt(user_config.ak, d)
    jwt_dict["refresh_time"] = get_current_formatted_time()
    user_token = UserToken(**jwt_dict)
    user_token.store_to_local()
    return user_token


def refresh_token(ak=None, sk=None, auth=False):
    user_config = get_config(ak, sk, auth=auth)
    if user_config is None:
        raise ValueError(
            "Local config must not be empty before refresh token. "
            "Please use the 'openxlab config' command to set the config"
        )
    local_user_token = get_token_from_local()
    if local_user_token is None:
        raise ValueError("Local token must not be empty before refresh token")
    refresh_expiration_datetime = get_datetime_from_formatted_str(
        local_user_token.refresh_expiration
    )
    now = get_current_time()
    if refresh_expiration_datetime <= now:
        return get_token_via_api(ak, sk)
    refresh_jwt_dict = AUTH_CLIENT.refresh_jwt(user_config.ak, local_user_token.refresh_token)
    refresh_jwt_dict["refresh_time"] = get_current_formatted_time()
    user_token = UserToken(**refresh_jwt_dict)
    user_token.store_to_local()
    return user_token


def get_installed_version() -> str:
    """
    Retrieves the version number of the currently installed openxlab.

    Returns:
        str: The version number as a string.

    Example Usage:
        >>> print(get_installed_version())
        '1.0.0'
    """
    return config_version.version


def update_package(use_official_index=False):
    """
    Upgrade the 'openxlab' package using pip.

    Args:
        use_official_index (bool, optional): Whether to use the official PyPI index. Defaults to False.

    Returns:
        None

    Raises exceptions:
        - An exception is issued if the package upgrade fails due to a subprocess error
        (e.g., network issues, invalid package name, etc.).
        - An exception is issued for any other unexpected exceptions that occur during the package upgrade process.
    """
    index_url = "https://pypi.org/simple" if use_official_index else None

    try:
        command = ["pip", "install", "--upgrade", "openxlab"]
        if index_url:
            command.extend(["--index-url", index_url])
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise EnvironmentError(
            f"An error occurred while installing or upgrading 'openxlab' using pip: {e}"
        )

    except Exception as e:
        raise Exception(
            f"An unknown exception occurred while installing or upgrading 'openxlab' using pip: {e}"
        )


def get_version_cache_local():
    """
    Retrieve local version.json content from the cache file.

    Returns:
        Optional[UserVersion]: The instantiated `UserVersion` object if loading was successful, or `None` otherwise.
    """
    try:
        version_json = get_file_content(get_version_path())
        version_dict = json.loads(version_json)
        return UserVersion(**version_dict)
    except Exception as e:
        print(f"Get the content of '~/.openxlab/version.json' failed. err: {e}")
        return None


def get_last_version_update_time():
    """
    Get the datetime of the last version cache update.

    Returns:
        Optional[str]: Datetime of the last cache update, or None if version.json does not exist.

    Raises:
        OSError: If there's an error accessing the file system. Do not handle.

    """
    try:
        if not os.path.exists(get_version_path()):
            return None
        version_cache_local = get_version_cache_local()
        if version_cache_local:
            return version_cache_local.last_version_check
    except Exception:
        pass
    return None


def get_version_cache_expiration(last_update_time: datetime) -> datetime:
    """
    Calculate the expiration time for cached data based on the last update time.

    Parameters:
        last_update_time (datetime): The date and time when the data is last refreshed.

    Returns:
        datetime: The calculated expiration time for the cache,
        which will be one day later than the last update time,
        and indicate when the cache becomes stale and needs refreshing again.
    """
    version_cache_expiration_time = last_update_time + timedelta(days=1)
    return version_cache_expiration_time


def trigger_update_check():
    """
    Triggers an update check if necessary based on the latest cache update time.

    Checks if the most recent cache update has expired according to predefined rules,
    and if so, performs an update check. Returns the new version found during the check,
    or None if no update check was needed or performed.

    Returns:
        Optional[str]: Version string of the new update found, or None if no update was checked/available.

    Steps:
        1. Retrieve the last update time from the cache.
        2. Convert the last update time into a datetime object.
        3. Calculate the expiration time for the cache based on the last update time.
        4. Compare the expiration time with the current time.
        5. If the cache is still valid (expiration time > now), do not perform an update check.
        6. Otherwise, initiate an update check and return the version of the update found, if any.
    """
    try:
        # Step 1: Get the last update time from version.json
        last_update_time = get_last_version_update_time()

        # Step 2: Convert the last update time into a datetime object
        if last_update_time is not None:
            last_update_datetime = get_datetime_from_formatted_str(last_update_time)

            # Step 3: Calculate the expiration time for the cache
            last_update_expiration_datetime = get_version_cache_expiration(last_update_datetime)

            # Step 4: Compare the expiration time with the current time
            now = get_current_time()
            if last_update_expiration_datetime > now:
                return None  # Step 5: Cache is still valid, no need for an update check

        # Step 6: Perform an update check since the cache is outdated
        current_version, latest_version, is_latest_version = update_version_check()

        # current_version < latest_version
        if not is_latest_version:
            print(
                f"Warning: you are using openxlab version {current_version}; "
                f"however, version {latest_version} is available. "
                f"Consider using 'pip install -U openxlab' to avoid version compatibility problems."
            )

    except Exception:
        pass


def update_version_check():
    """
    Check the current installed version against the latest one, possibly prompting for an update.

    Steps:
        1. Fetch the currently installed version of the application.
        2. Communicate with the remote service to obtain the latest version status.
           The response includes flags for automatic updates and the availability of a newer version.
        3. If the auto-update flag is set and the current version is older than the latest,
           initiate the update process and inform the user of the upgrade.
        4. Regardless of an update, record the current datetime and store the latest version information.
        5. Return the pair of the current and latest version numbers.
    """
    try:
        # step1. Fetch the currently installed version of the application
        current_version = get_installed_version()

        # step2. Obtain the latest version information via api
        auto_update, is_latest_version, latest_version = VERSION_CLIENT.version_check(
            version=current_version
        )

        # step3.Forced update when auto_update=True and is_latest_version=False, it means current version is outdated
        if auto_update and not is_latest_version:
            update_package(use_official_index=True)
        # get current time
        now = get_current_formatted_time()

        # save latest version information to local version.json
        latest_version_obj = LatestVersion(
            is_latest_version=is_latest_version,
            latest_version=latest_version,
            auto_update=auto_update,
        )
        latest_version_dict = latest_version_obj.__dict__

        version_info = UserVersion(last_version_check=now, latest_version_data=latest_version_dict)

        version_info.store_to_local()
        return current_version, latest_version, is_latest_version
    except Exception as e:
        raise Exception(f"Update check failed. {e}") from e


class UserToken(object):
    def __init__(self, jwt, expiration, sso_uid, refresh_time, refresh_token, refresh_expiration):
        self.jwt = jwt
        self.expiration = expiration
        self.sso_uid = sso_uid
        self.refresh_time = refresh_time
        self.refresh_token = refresh_token
        self.refresh_expiration = refresh_expiration

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def store_to_local(self):
        if not os.path.exists(get_config_dir()):
            os.makedirs(get_config_dir(), mode=0o700)
        token_json = self.to_json()
        with open(get_token_path(), mode="w", encoding="utf-8") as f:
            f.write(token_json)


class UserVersion(object):
    def __init__(self, last_version_check, latest_version_data) -> None:
        """

        Parameters
        ----------
        last_version_check : _type_
            2024-04-28 16:52:20
        latest_version_data : _type_
            "{
                "is_latest_version": bool,
                "latest_version": "",
                "auto_update": bool
            }"
        """
        self.last_version_check = last_version_check
        self.latest_version_data = latest_version_data

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def store_to_local(self):
        if not os.path.exists(get_config_dir()):
            os.makedirs(get_config_dir(), mode=0o700)
        cache_json = self.to_json()
        lock_path = get_version_path() + ".lock"
        # blocking=False means non-blocking, it is prior than timeout.
        # Set timeout=-1 can make update check and write version.json as a atomic operation
        lock = FileLock(lock_file=lock_path, blocking=False)
        try:
            with lock:
                with open(get_version_path(), mode="w", encoding="utf-8") as f:
                    f.write(cache_json)
        except Exception as e:
            raise Exception(
                f"An error occurred while writing to ~/.openxlab/version.json file: {e}"
            ) from e


class LatestVersion(object):
    """
    A class to represent the latest version information of a software package.

    Attributes:
        is_latest_version (bool): Indicates whether the current installed version is the latest one.
        latest_version (str): The version string of the latest available release.
        auto_update (bool): Specifies if automatic updates are enabled for the package.

    Methods:
        to_json(): Converts the object to a JSON-formatted string for easy serialization.

    Usage:
        Instantiate the class with the current version status, latest version, and auto-update setting:

        latest_info = LatestVersion(is_latest_version, latest_version, auto_update)

        Convert the object to a JSON string:

        json_string = latest_info.to_json()
    """

    def __init__(self, is_latest_version, latest_version, auto_update) -> None:
        self.is_latest_version = is_latest_version
        self.latest_version = latest_version
        self.auto_update = auto_update

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
