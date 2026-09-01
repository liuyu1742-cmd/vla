import unittest


class OpenVlaIpcPredictTests(unittest.TestCase):
    def test_request_requires_image_path(self):
        from tools.openvla_ipc_predict import validate_request

        with self.assertRaisesRegex(ValueError, "image_path"):
            validate_request({"instruction": "pick up the cup"})
