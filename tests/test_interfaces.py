import unittest

import numpy as np
from PIL import Image

import collector_vision as cvg
from collector_vision.interfaces import DetectionResult


class DetectionResultTests(unittest.TestCase):
    def test_dewarp_outputs_embedder_sized_square_crop(self) -> None:
        bgr = np.zeros((60, 80, 3), dtype=np.uint8)
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        crop = DetectionResult(corners=corners, card_present=True).dewarp(bgr)

        self.assertEqual(crop.size, (448, 448))


class TransformTests(unittest.TestCase):
    def test_rotate_card_180_flips_pixels(self) -> None:
        crop = Image.new("RGB", (2, 2))
        crop.putpixel((0, 0), (255, 0, 0))
        crop.putpixel((1, 0), (0, 255, 0))
        crop.putpixel((0, 1), (0, 0, 255))
        crop.putpixel((1, 1), (255, 255, 255))

        rotated = cvg.rotate_card_180(crop)

        self.assertEqual(rotated.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(rotated.getpixel((1, 0)), (0, 0, 255))
        self.assertEqual(rotated.getpixel((0, 1)), (0, 255, 0))
        self.assertEqual(rotated.getpixel((1, 1)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
