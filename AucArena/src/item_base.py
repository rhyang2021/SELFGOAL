import sys
sys.path.append('..')
from utils import LoadJsonL


class Item():
    def __init__(self, id: int, name: str, price: int, desc: str, true_value: int):
        self.id = id
        self.name = name
        self.price = price
        self.desc = desc
        self.true_value = true_value
        self._original_price = price

    def get_desc(self):
        return f"{self.name}, starting at ${int(self.price)}."

    def __repr__(self):
        return f"{self.name}"
    
    def __str__(self):
        return f"{self.name}"
    
    def info(self):
        return f"{self.name}: ${int(self.price)} to ${self.true_value}."

    def lower_price(self, percentage: float = 0.2):
        # lower starting price by 20%
        self.price = int(self.price * (1 - percentage))
    
    def reset_price(self):
        self.price = self._original_price


def create_items(item_info_jsl):
    '''
    item_info: a list of dict (name, price, desc, id)
    '''
    item_info_jsl = LoadJsonL(item_info_jsl)
    item_list = []
    for info in item_info_jsl:
        item_list.append(Item(**info))
    return item_list


def item_list_equal(items_1: list, items_2: list):
    # could be a list of strings (names) or a list of Items
    item_1_names = [item.name if isinstance(item, Item) else item for item in items_1]
    item_2_names = [item.name if isinstance(item, Item) else item for item in items_2]
    return set(item_1_names) == set(item_2_names)