import unittest

from tests.mape_test import MAPETestCase

from tests.validator_test import ValidatorTestCase


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(MAPETestCase())
    test_suite.addTest(ValidatorTestCase('test_widget_resize'))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
