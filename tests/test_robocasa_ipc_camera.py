import unittest


class RoboCasaIpcCameraTests(unittest.TestCase):
    def test_camera_request_path_is_png(self):
        from tools.robocasa_ipc_camera import image_path_for_step

        self.assertEqual(image_path_for_step(3).name, "frame_0003.png")
