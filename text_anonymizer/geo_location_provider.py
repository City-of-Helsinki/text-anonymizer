import csv
import os
import sys
from string import digits
from xml.dom import minidom


def get_data_file_path(file_name):
    bundle_dir = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(bundle_dir, "data", file_name)


def load_city_data(file_path):
    with open(file_path, 'r') as data:
        return [line[1].lower() for line in csv.reader(data, delimiter=';')][1:]


def load_post_office_data(file_path):
    with minidom.parse(file_path) as po_data:
        post_offices = po_data.getElementsByTagName('toimipaikka')
        remove_digits = str.maketrans('', '', digits)
        locations = []
        for p in post_offices:
            n_elem = p.getElementsByTagName('nimi')[0]
            end_date_elem = p.getElementsByTagName('lakkauttamispäivämäärä')[0]
            if not end_date_elem.firstChild and n_elem.firstChild:
                name_parts = n_elem.firstChild.data.split(' - ')
                for n in name_parts:
                    n = n.translate(remove_digits).strip().lower()
                    if n not in locations:
                        locations.append(n)
        return locations


_ALL_LOCATIONS = load_city_data(get_data_file_path("kuntaluettelo-suppeat-tiedot-2021-01-01.csv"))
_ALL_LOCATIONS += load_post_office_data(get_data_file_path("suomen_postitoimipaikat.xml"))
_ALL_LOCATIONS = [loc for loc in _ALL_LOCATIONS if loc not in ['ii', 'eno']]


def get_location_names():
    return _ALL_LOCATIONS