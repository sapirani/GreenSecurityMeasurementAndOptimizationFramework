import unittest

from utils.general_functions import EnvironmentImpact


class Convert_Mwh(unittest.TestCase):
    def test_convert_mwh_to_co2(self):
        self.assertAlmostEqual(EnvironmentImpact.from_mwh(1e6).co2, 0.709)

    def test_convert_mwh_to_coal_burned(self):
        self.assertAlmostEqual(EnvironmentImpact.from_mwh(1e6).coal_burned, 0.784 * 0.453592)

    def test_convert_mwh_to_number_of_smartphones_charged(self):
        self.assertAlmostEqual(EnvironmentImpact.from_mwh(1e6).number_of_smartphones_charged, 86.2)

    def test_convert_mwh_to_kg_of_woods_burned(self):
        self.assertAlmostEqual(EnvironmentImpact.from_mwh(3.5 * 1e6).kg_of_woods_burned, 1)
