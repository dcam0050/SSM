import pytest


@pytest.fixture(scope="module")
def get_error_list(request):
    print "setting up stuff"
    error_list = []

    def teardown():
        print "ERROR LIST", error_list
        ind = range(len(error_list))
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(ind, error_list, color='r')
        # ax.set_ylabel('Average % error / class')
        # ax.set_title('Error across test scenarios')
        # plt.show()
    request.addfinalizer(teardown)
    return error_list
