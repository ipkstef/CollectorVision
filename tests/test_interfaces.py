import unittest

import numpy as np
from PIL import Image

import collector_vision as cvg
from collector_vision.detectors.neural import _order_corners
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


class NeuralDetectorCornerOrderingTests(unittest.TestCase):
    def test_shortest_edge_orientation_uses_original_image_space(self) -> None:
        corners = np.array(
            [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]],
            dtype=np.float32,
        )

        ordered = _order_corners(corners, image_shape=(500, 2000, 3))

        np.testing.assert_allclose(
            ordered,
            np.array(
                [[0.6, 0.2], [0.6, 0.8], [0.4, 0.8], [0.4, 0.2]],
                dtype=np.float32,
            ),
        )

    def test_shortest_edge_becomes_top_from_any_side(self) -> None:
        corners = np.array(
            [[0.0, 0.0], [0.8, 0.0], [0.7, 0.8], [0.0, 0.8]],
            dtype=np.float32,
        )

        ordered = _order_corners(corners, image_shape=(1000, 1000, 3))

        np.testing.assert_allclose(
            ordered,
            np.array(
                [[0.7, 0.8], [0.0, 0.8], [0.0, 0.0], [0.8, 0.0]],
                dtype=np.float32,
            ),
        )

    def test_portrait_orientation_keeps_short_edge_on_top(self) -> None:
        corners = np.array(
            [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]],
            dtype=np.float32,
        )

        ordered = _order_corners(corners, image_shape=(2000, 500, 3))

        np.testing.assert_allclose(ordered, corners)


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
