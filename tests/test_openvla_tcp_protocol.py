import unittest

from tools.openvla_tcp_protocol import decode_message, encode_message


class OpenVLATcpProtocolTests(unittest.TestCase):
    def test_round_trip_is_one_json_line(self):
        encoded = encode_message({"instruction": "pick up the glass cup"})
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(decode_message(encoded), {"instruction": "pick up the glass cup"})

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_message(b"\n")


if __name__ == "__main__":
    unittest.main()
