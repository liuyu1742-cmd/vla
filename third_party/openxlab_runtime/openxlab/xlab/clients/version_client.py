import re

import requests as requests


def http_common_header():
    header_dict = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    return header_dict


class VersionClient(object):
    """
    A class representing a client for interacting with a version checking service over HTTP. Don't use authorization.

    Usage:
        client = VersionClient(endpoint_url)
        result = client.version_check(current_version)
        print(result)
    """

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def validate_response(self, response_dto, path, payload):
        if response_dto['msg_code'] != '10000':
            raise ValueError(
                f"call {path} error, payload: {payload}, message: {response_dto['msg']}"
            )
        response_data = response_dto["data"]
        if response_data is None or len(response_data) == 0:
            raise ValueError(
                f"call {path} error, payload: {payload}, message: {response_dto['msg']}"
            )
        return response_data

    def from_camel_case(self, camel_dict: dict):
        snake_dict = {}
        for key, value in camel_dict.items():
            snake_key = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
            snake_dict[snake_key] = value
        return snake_dict

    def http_post_response_dto(self, path, payload):
        headers = http_common_header()
        response = requests.post(f"{self.endpoint}{path}", json=payload, headers=headers)
        response.raise_for_status()
        response_dict = response.json()
        response_dto = self.from_camel_case(response_dict)
        return response_dto

    def version_check(self, version: str):
        """
        get the latest version information via api.

        response_dto structure:
        {
            "code": null,
            "msgCode": "10000",
            "msg": "Success",
            "message": null,
            "success": true,
            "data": {
                "is_latest_version": false,
                "latest_version": "OpenXLab 0.0.37",
                "auto_update": false
            }
        }
        """
        path = "api/v1/cli/version"
        payload = {"current_version": version}
        response_dto = self.http_post_response_dto(path, payload)
        version_dict = self.validate_response(response_dto, path, payload)

        # get the latest version
        latest_version = version_dict["latest_version"].split()[-1]
        return (
            version_dict["auto_update"],
            version_dict["is_latest_version"],
            latest_version,
        )
