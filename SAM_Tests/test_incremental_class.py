import pytest
import pytest_dependency

class sm:
    def __init__(self):
        self.val = None
        self.val2 = None


df = sm()

combs = [True, False, True]
bombs = ['this', 'that']


class TestSampleWithScenarios(object):
    scenario_keys = ['attribute', 'att2']
    scenario_parameters = [combs, bombs]

    def test_train_demo(self, attribute, att2):
        print attribute, att2
        assert True


    def test_load_demo(self, attribute2):
        df.val2 = attribute2
        assert attribute2

    def test_respond_demo(self, attribute2):
        assert True

    def test_close_demo(self, attribute2):
        assert True
