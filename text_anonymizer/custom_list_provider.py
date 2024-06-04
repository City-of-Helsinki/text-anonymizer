import os
import sys


def get_data_file_path(file_name):
    bundle_dir = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(bundle_dir, "config", file_name)


def load_list_data(file_path):
    with open(file_path, 'r') as file:
        return [line.lower().replace("\n", "") for line in file.readlines() if not line.startswith("#")]


_BLOCK_LIST = load_list_data(get_data_file_path("blocklist.txt"))
_GRANT_LIST = load_list_data(get_data_file_path("grantlist.txt"))


def get_grant_list():
    return _GRANT_LIST


def get_block_list():
    return _BLOCK_LIST