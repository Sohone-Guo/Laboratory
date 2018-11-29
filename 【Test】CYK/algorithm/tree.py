

class BTree():
    """

    # Arguments
        result {dic}: e.g.
            {
                (0, 0): {'a': ['C ']}, 
                (1, 0): {'a': ['C ']}, 
                (2, 0): {'a': ['C ']}, 
                (3, 0): {'b': ['D ']}, 
                (4, 0): {'b': ['D ']}, 
                (5, 0): {'b': ['D ']}, 
                (6, 0): {'c': ['B ', 'E ']}, 
                (7, 0): {'c': ['B ', 'E ']}, 
                (2, 1): {((2, 0), (3, 0)): ['A ']}, 
                (6, 1): {((6, 0), (7, 0)): ['B ']}, 
                (2, 2): {((2, 1), (4, 0)): ['F ']}, 
                (1, 3): {((1, 0), (2, 2)): ['A ']}, 
                (1, 4): {((1, 3), (5, 0)): ['F ']}, 
                (0, 5): {((0, 0), (1, 4)): ['A ']}, 
                (0, 6): {((0, 5), (6, 0)): ['S ']}, 
                (0, 7): {((0, 5), (6, 1)): ['S ']}
            }
    
    # Returns
        parsing {str}
    """
    def __init__(self, data):
        self.data = data
        self.max_index = self.max_index_check(data) 

        self.build_tree()
    
    def max_index_check(self, data):
        """
        # Arguments
            data {dic}

        # Return
            current {int}

        """
        current = 0
        for key in data:
            if key[1] > current:
                current = key[1]
        return current

    def build_tree(self):
        root = self.data[(0, self.max_index)]

        def recurse(node):
            for key in node:
                position = key
                value = node[key]
            if isinstance(position,str):
                return (position, value)
            return ("({}, ({}, {}))".format(value, recurse(self.data[position[0]]), recurse(self.data[position[1]])))
            
        return recurse(root)


if __name__ == "__main__":
    data = {
                (0, 0): {'a': ['C ']}, 
                (1, 0): {'a': ['C ']}, 
                (2, 0): {'a': ['C ']}, 
                (3, 0): {'b': ['D ']}, 
                (4, 0): {'b': ['D ']}, 
                (5, 0): {'b': ['D ']}, 
                (6, 0): {'c': ['B ', 'E ']}, 
                (7, 0): {'c': ['B ', 'E ']}, 
                (2, 1): {((2, 0), (3, 0)): ['A ']}, 
                (6, 1): {((6, 0), (7, 0)): ['B ']}, 
                (2, 2): {((2, 1), (4, 0)): ['F ']}, 
                (1, 3): {((1, 0), (2, 2)): ['A ']}, 
                (1, 4): {((1, 3), (5, 0)): ['F ']}, 
                (0, 5): {((0, 0), (1, 4)): ['A ']}, 
                (0, 6): {((0, 5), (6, 0)): ['S ']}, 
                (0, 7): {((0, 5), (6, 1)): ['S ']}
            }

    BTree(data)