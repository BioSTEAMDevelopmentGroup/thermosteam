import sys

def pytest_ignore_collect(path):
    path = str(path)
    if 'setup' in path:
        return True
