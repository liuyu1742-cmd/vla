import os
import sys

import oss2
from oss2 import SizedFileAdapter
from oss2.exceptions import NoSuchUpload
from tqdm import tqdm

from openxlab.model.common.constants import oss_accelerate_domain
from openxlab.model.common.constants import oss_bucket_domain
from openxlab.model.common.constants import oss_endpoint_domain
from openxlab.utils.local_cache import cache


class OssClient(object):
    def __init__(self, endpoint, access_key_id, access_key_secret, security_token, bucket_name):
        self.bucket = self.get_bucket(
            endpoint, access_key_id, access_key_secret, security_token, bucket_name
        )
        self.cache_upload_id_key_prefix = "model:oss:upload_id"

    def get_bucket(self, endpoint, access_key_id, access_key_secret, security_token, bucket_name):
        auth = oss2.StsAuth(access_key_id, access_key_secret, security_token)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
        return self.bucket

    def upload_to_oss(self, local_file_path, object_key, callback, callback_var):
        """
        simple upload
        """
        headers = {
            "Host": oss_bucket_domain,
            "x-oss-callback": callback,
            "x-oss-callback-var": callback_var,
        }
        self.bucket.put_object_from_file(
            object_key, local_file_path, headers, progress_callback=percentage
        )

    def multipart_resume_upload(
        self, local_file_path, object_key, callback=None, callback_var=None, domain=None
    ):
        headers = {"x-oss-callback": callback, "x-oss-callback-var": callback_var}
        if domain == "endpoint":
            headers["Host"] = oss_endpoint_domain
        elif domain == "bucket":
            headers["Host"] = oss_bucket_domain
        elif domain == "accelerate":
            headers["Host"] = oss_accelerate_domain

        total_size = os.path.getsize(local_file_path)
        part_size = oss2.determine_part_size(total_size, preferred_size=100 * 1024)
        upload_id = None
        cache_upload_id_key = f"{self.cache_upload_id_key_prefix}:{object_key}"
        try:
            upload_id = cache.get(cache_upload_id_key)
            uploaded_parts_numbers = []
            uploaded_parts = []
            if upload_id:
                uploaded_parts_result = self.bucket.list_parts(object_key, upload_id)
                if uploaded_parts_result:
                    uploaded_parts = uploaded_parts_result.parts
                    uploaded_parts_numbers = [part.part_number for part in uploaded_parts]
            else:
                # init upload
                upload_id = self.bucket.init_multipart_upload(object_key).upload_id
                cache.set(cache_upload_id_key, upload_id, expire=3600 * 24 * 7)
            # compute part count
            part_count = (total_size + part_size - 1) // part_size

            # begin multipart upload
            parts = [] + uploaded_parts
            with open(local_file_path, "rb") as fileobj:
                progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
                for i in range(part_count):
                    part_number = i + 1
                    if part_number in uploaded_parts_numbers:
                        continue
                    offset = i * part_size
                    num_to_upload = min(part_size, total_size - offset)
                    upload_part = self.bucket.upload_part(
                        object_key,
                        upload_id,
                        part_number,
                        SizedFileAdapter(fileobj, num_to_upload),
                    )
                    parts.append(oss2.models.PartInfo(part_number, upload_part.etag))
                    progress_bar.update(offset + num_to_upload - progress_bar.n)
            progress_bar.close()
            # complete multipart upload
            self.bucket.complete_multipart_upload(object_key, upload_id, parts, headers)
            cache.delete(cache_upload_id_key)
            print(f"Successfully uploaded {object_key}")
        except Exception as e:
            print(f"Error uploading {object_key}: {e}")
            if isinstance(e, NoSuchUpload):
                # cancel upload task if error
                cache.delete(cache_upload_id_key)
                self.bucket.abort_multipart_upload(object_key, upload_id)
                print(f"Upload of {object_key} aborted")
            raise e


# consumed_bytes表示已上传的数据量。
# total_bytes表示待上传的总数据量。当无法确定待上传的数据长度时，total_bytes的值为None。
def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print(f"{consumed_bytes}/{total_bytes} |  {rate}% has been uploaded!")
        sys.stdout.flush()
