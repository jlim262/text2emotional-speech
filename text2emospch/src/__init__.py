import pkgutil
import sys
import os


def load_all_modules_from_dir(dirname):
    for importer, package_name, _ in pkgutil.iter_modules([dirname]):
        if package_name not in sys.modules and package_name != 'main':
            module = importer.find_module(
                package_name).load_module(package_name)
            print(f'{module} is loaded')


root_dir = os.path.dirname(__file__)
load_all_modules_from_dir(root_dir)
print(root_dir)

for root, subdirs, files in os.walk(root_dir):
    ignore = []  # ['optimizer']
    for sub in subdirs:
        if sub in ignore:
            continue

        cur = os.path.join(root, sub)
        if os.path.isdir(cur):
            load_all_modules_from_dir(cur)
