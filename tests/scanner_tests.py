import unittest

from Scanner.scanner import convert_mwh_to_other_metrics


class Convert_Mwh(unittest.TestCase):
    def test_convert_mwh_to_co2(self):
        self.assertAlmostEqual(convert_mwh_to_other_metrics(1e6)[0], 0.709)  # add assertion here

    def test_convert_mwh_to_coal_burned(self):
        self.assertAlmostEqual(convert_mwh_to_other_metrics(1e6)[1], 0.784 * 0.453592)  # add assertion here

    def test_convert_mwh_to_number_of_smartphones_charged(self):
        self.assertAlmostEqual(convert_mwh_to_other_metrics(1e6)[2], 86.2)  # add assertion here

    def test_convert_mwh_to_kg_of_woods_burned(self):
        self.assertAlmostEqual(convert_mwh_to_other_metrics(3.5 * 1e6)[3], 1)  # add assertion here


if __name__ == '__main__':
    unittest.main()
