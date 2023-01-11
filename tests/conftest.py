import tests.luxai_s2_patch


def pytest_runtest_setup(item):
    tests.luxai_s2_patch.install_patches()
