README_FILE_NAME = "README.md"
FILE_THRESHOLD = 30 * 1024 * 1024
# FILE_THRESHOLD = 100
# connection timeout and waiting response timeout
TIMEOUT = (30, None)
UPLOAD_TIMEOUT = (60, None)
# max retries when encounter connectionerror
MAX_RETRIES = 5
BASE_DELAY = 5
MAX_DELAY = 32

# upload
CKPT_SUFFIX = ".oxl"
CKPT_FOLDER = ".cache_oxl"
MIN_FILE_SEGMENTATION = 100 * 1024 * 1024  # 100M

# maximum number of files to show per directory in the info command
MAX_FILES_PER_DIR = 8


# # staging
# odl_clientId = "ypkl8bwo0eb5ao1b96no"
# endpoint = "https://staging.openxlab.org.cn"
# uaa_url_prefix = "https://sso.staging.openxlab.org.cn/gw/uaa-be"

# prod
odl_clientId = "kmz3bkwzlaa3wrq8pvwa"
endpoint = "https://openxlab.org.cn"
uaa_url_prefix = "https://sso.openxlab.org.cn/gw/uaa-be"


computed_url = "/datasets/api/v2/"
