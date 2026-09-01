import json
import sys
from typing import Dict
from urllib.parse import quote

import requests
from rich import print as rprint

from openxlab.dataset.constants import BASE_DELAY
from openxlab.dataset.constants import computed_url
from openxlab.dataset.constants import MAX_DELAY
from openxlab.dataset.constants import MAX_RETRIES
from openxlab.dataset.constants import TIMEOUT
from openxlab.dataset.exception import OpenDataLabError
from openxlab.dataset.utils import highlight_urls
from openxlab.dataset.utils import retry_with_backoff
from openxlab.xlab.handler.user_token import get_jwt


base_headers = {
    "accept": "application/json",
}
headers_post_base = {
    **base_headers,
    "Content-Type": "application/json",
}


class XlabDatasetAPI(object):
    """This class contains the interaction between client & odl serverend.

    This class is being instantiate from client.py
    Strongly recommend that openxlab provide a general method to handle contextinfo.

    """

    def __init__(self, host, cookie):
        self.host = host
        self.odl_cookie = cookie

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def get_dataset_files(
        self, dataset_name: str, payload: dict = None, needContent=True, auth=True, after: str = None, limit: int = 500
    ):
        """
        get files info list

        Parameters
        ----------
        dataset_name : str
            the name of dataset
        payload : dict, optional
            by default None
        needContent : bool, optional
            if need info like size etc, set True, by default False
        auth : bool, optional
            enable authorization, set True, by default True.
        after : str, optional
            pagination cursor for next page, by default None (start from beginning)
        limit : int, optional
            number of items per page, by default 1000
        Returns
        -------
        """
        # get and download need authorization
        if auth:
            auth_headers = self.http_authorization_header()
            headers = auth_headers
        # list do not need authorization
        else:
            headers = base_headers
        data = {"recursive": True, "needContent": needContent, "limit": limit}
        
        # Add after cursor if provided
        if after:
            data["after"] = after

        source_path = "/"
        if payload:
            data.update(payload)
            source_path = payload["prefix"]

        resp = requests.get(
            url=f"{self.host}/datasets/api/v2/datasets/{dataset_name}/r/main",
            params=data,
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            rprint(f"{OpenDataLabError(resp.status_code, highlight_urls(resp.text))}")
            sys.exit(-1)
        result_dict = resp.json()['data']
        # Any other files besides readme.md and metafile.yaml?
        if not result_dict['hasMediaFile']:
            rprint(f"{OpenDataLabError(error_msg=highlight_urls(str(result_dict['toast'])))} ")
            sys.exit(-1)
        # no content
        if not result_dict['list']:
            err_msg = OpenDataLabError(
                404,
                f'Failed to retrieve the dataset list. Please verify if the source path "{source_path}" '\
                'exists within the dataset repository.',
            )
            print(err_msg)
            sys.exit(-1)
        return result_dict

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def download_check(self, dataset_id: str, path: str):
        """check file when download it for crawler"""
        headers = self.http_authorization_header()

        resp = requests.get(
            url=f"{self.host}/datasets/api/v2/downloadCheck/{dataset_id}/main/{path}",
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )

        # add special symbol for web when the user do not fill in the form of user information
        # or the form of application for dataset
        if resp.status_code == 601 or resp.status_code == 602:
            err_msg = OpenDataLabError(
                resp.status_code, highlight_urls(text=resp.text, suffix="?from=cli")
            )
            rprint(f"{err_msg}")
            sys.exit(-1)

        if resp.status_code != 200:
            rprint(f"{OpenDataLabError(resp.status_code, highlight_urls(resp.text))}")
            sys.exit(-1)
        return

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def get_dataset_download_urls(self, dataset_id: int, dataset_dict: Dict):
        headers = self.http_authorization_header()

        resp = requests.get(
            url=f"{self.host}/datasets/resolve/{dataset_id}/main/{dataset_dict['name']}",
            headers=headers,
            cookies=self.odl_cookie,
            allow_redirects=False,
            timeout=TIMEOUT,
        )
        if resp.status_code != 302:
            print(f"{OpenDataLabError(resp.status_code, resp.text)}")
            sys.exit(-1)
        return resp.headers['Location']

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def get_dataset_info(self, dataset_name: str):
        parsed_dataset_name = quote(dataset_name.replace("/", ","))
        resp = requests.get(
            url=f"{self.host}{computed_url}datasets/{parsed_dataset_name}",
            headers=base_headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"{OpenDataLabError(resp.status_code, resp.text)}")
            sys.exit(-1)

        data = resp.json()['data']
        if data['id'] == 0:
            print(f"No dataset:{dataset_name}")
            sys.exit(-1)
        return data

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def pre_object_upload(self, dataset: str, branch: str, file_path: str, req: dict) -> dict:
        # optimize
        headers = self.http_authorization_header()
        if file_path.startswith('/'):
            file_path = file_path[1:]
        resp = requests.post(
            url=f"{self.host}{computed_url}preUpload/{dataset}/{branch}/{file_path}",
            data=json.dumps(req),
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise Exception(f"{OpenDataLabError(resp.status_code, resp.text)}")
        resp_json = resp.json()
        data = resp_json['data']
        return data

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def post_object_upload(self, dataset: str, branch: str, file_path: str, req: dict) -> dict:
        headers = self.http_authorization_header()
        # optimize
        if file_path.startswith('/'):
            file_path = file_path.split('/', 1)[-1]
        resp = requests.post(
            url=f"{self.host}{computed_url}postUpload/{dataset}/{branch}/{file_path}",
            data=json.dumps(req),
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"{OpenDataLabError(resp.status_code, resp.text)}")
        resp_json = resp.json()
        data = resp_json["data"]
        return data

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def create_dataset(self, req: dict):
        headers = self.http_authorization_header()
        resp = requests.post(
            url=f"{self.host}/datasets/api/v2/datasets",
            data=json.dumps(req),
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise Exception(
                f"While creating {req['name']} encounter error: {OpenDataLabError(resp.text)}"
            )
        resp_json = resp.json()
        data = resp_json['data']
        return data

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def commit_dataset(self, req: list):
        dataset_name = req[0]
        parsed_dataset_name = quote(dataset_name.replace("/", ","))

        headers = self.http_authorization_header()
        resp = requests.post(
            url=f"{self.host}/datasets/api/v2/datasets/{parsed_dataset_name}/commit",
            data=json.dumps(req[1]),
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"While committing {req[0]} encounter error: {OpenDataLabError(resp.text)}")
            sys.exit(-1)
        return

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def delete_repo(self, dataset_repo_name: str):
        """delete repo api"""
        headers = self.http_authorization_header()
        resp = requests.delete(
            url=f"{self.host}{computed_url}datasets/{dataset_repo_name}",
            cookies=self.odl_cookie,
            headers=headers,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            dataset_repo_name = dataset_repo_name.replace(",", "/")
            print(
                f"While deleting {dataset_repo_name} encounter error: {OpenDataLabError(resp.text)}"
            )
            sys.exit(-1)
        return

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def set_repo_permission(self, dataset_repo_name: str, private: bool):
        """
        change repo permission
        @private: true is private, false is public
        """
        permission = "private" if private else "public"
        headers = self.http_authorization_header()

        resp = requests.post(
            url=f"{self.host}{computed_url}datasets/{dataset_repo_name}/actions/changeState?state={permission}",
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            ds_name = dataset_repo_name.replace(",", "/")
            print(
                f"while change {ds_name} visibility encounter error:{OpenDataLabError(resp.text)}"
            )
            sys.exit(-1)
        return

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def track_query_dataset_files(self, dataset_name: str):
        """track the operation of getting dataset files"""
        resp = requests.post(
            url=f"{self.host}/datasets/api/v2/track/datasets/cli/ls/{dataset_name}",
            headers=base_headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"{OpenDataLabError(resp.status_code, resp.text)}")
            sys.exit(-1)
        return

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def track_download_dataset_files(self, dataset_name: str, file_path: str):
        """track the operation of downloading dataset files"""
        headers = self.http_authorization_header()
        body = {"path": file_path}
        resp = requests.post(
            url=f"{self.host}/datasets/api/v2/track/datasets/cli/download/{dataset_name}",
            headers=headers,
            cookies=self.odl_cookie,
            data=body,
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"{OpenDataLabError(resp.status_code, resp.text)}")
            sys.exit(-1)
        return

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def check_public_validation(
        self,
        dataset_name: str,
    ):
        """check dataset repository can be set public or not"""
        headers = self.http_authorization_header()

        resp = requests.post(
            url=f"{self.host}/datasets/api/v2/datasets/{dataset_name}/actions/validate",
            headers=headers,
            cookies=self.odl_cookie,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            print(
                f"While check public validation of {dataset_name} encounter error: {OpenDataLabError(resp.text)}"
            )
            sys.exit(-1)
        resp_json = resp.json()
        data = resp_json['data']
        return data

    @retry_with_backoff(max_retries=MAX_RETRIES, base_delay=BASE_DELAY, max_delay=MAX_DELAY)
    def moderate_text(self, content: str, text_type: int):
        """
        scan and verify text, ensure it's green and does not contain offensive, inappropriate or illegal content

        Parameters
        ----------
        content : str
            the text to scan and moderate
        text_type : int
            the type of moderation. if only moderate readme.md, type=3
        """
        headers = self.http_authorization_header()

        body = {"content": content, "type": text_type}
        resp = requests.post(
            url=f"{self.host}/datasets/api/v2/datasets/textAudit",
            headers=headers,
            cookies=self.odl_cookie,
            data=json.dumps(body),
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"{OpenDataLabError(resp.text)}")
            return False
        return True

    def http_authorization_header(self, header_dict=headers_post_base) -> dict:
        """
        Generate and add Authorization header with JWT to the given header dictionary.

        Args:
            header_dict (dict, optional): A dictionary of HTTP headers. Defaults to headers_post_base.

        Returns:
            dict: The updated header dictionary with the Authorization header added.
        """

        try:
            jwt = get_jwt(auth=True)
            header_dict['Authorization'] = jwt
        except Exception as e:
            # raise Exception(f"{e}")
            print(f"{e}")
            sys.exit(-1)
        return header_dict
