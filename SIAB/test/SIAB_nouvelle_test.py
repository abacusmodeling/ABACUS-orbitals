import SIAB.SIAB_nouvelle as siab
import unittest
from unittest.mock import patch
import argparse

class TestSIABNouvelle(unittest.TestCase):

    # unittest for initialize is written by Chat-AISI
    def test_initialize_defaults(self):
        """Test the function with default values."""
        input_script, is_test, version = siab.initialize(False)  # `False` to avoid using the command line args
        
        self.assertEqual(input_script, "./SIAB_INPUT")
        self.assertFalse(is_test)
        self.assertEqual(version, "0.1.0")

    @patch('argparse.ArgumentParser.parse_args')
    def test_initialize_with_args(self, mock_args):
        """Test the function with command line arguments."""
        # Set up the mock_args to simulate command line arguments.
        mock_args.return_value = argparse.Namespace(input='./Another_SIAB_INPUT', test=True, version='2.0.0')

        input_script, is_test, version = siab.initialize(True)
        
        self.assertEqual(input_script, './Another_SIAB_INPUT')
        self.assertTrue(is_test)
        self.assertEqual(version, '2.0.0')

# Running the tests
if __name__ == "__main__":
    unittest.main()
