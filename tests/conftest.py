import tests.luxai2022_patch


def pytest_runtest_setup(item):
    tests.luxai2022_patch.install_patches()
