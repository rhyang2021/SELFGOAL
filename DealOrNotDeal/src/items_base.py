import sys

sys.path.append('..')
from utils import LoadJsonL


class Item():
    def __init__(self, count, value):
        self.count = count
        self.value = value

    def info(self):
        return self.count, self.value

    def __repr__(self):
        return f"Item(count={self.count}, value={self.value})"

    def __str__(self):
        return f"Item with count {self.count} and value {self.value}"


def create_items(item_info_jsl):
    '''
    item_info: a list of dict (name, price, desc, id)
    '''
    item_info_jsl = LoadJsonL(item_info_jsl)
    items_list = []
    for info in item_info_jsl:
        items_list.append(Item(**info))
    return items_list
