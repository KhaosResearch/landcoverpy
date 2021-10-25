import unittest

import numpy as np

from composite import composite

class TestComposite(unittest.TestCase):
    def test_composite_mean(self):
        composite_out = composite("tests/data/B02_60m.jp2", "tests/data/B03_60m.jp2", "tests/data/B04_60m.jp2", method="mean")

    def test_composite_median(self):
        composite_out = composite("tests/data/B02_60m.jp2", "tests/data/B03_60m.jp2", "tests/data/B04_60m.jp2", method="median")
        

    def test_wrong_sizes(self):
        with self.assertRaises(ValueError) as context:
            composite("tests/data/B02_60m.jp2", "tests/data/B02_10m.jp2", method="median")

        self.assertTrue('Not all bands have the same shape' in str(context.exception))
