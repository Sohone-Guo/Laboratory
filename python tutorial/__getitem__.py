import collections 

Card = collections.namedtuple("Card", ["rank", "suit"])

class frenchDesk:
    ranks = [str(n) for n in range(2, 11)]+list("JQKA")
    suits = "spades diamonds clubs hearts".split()
    
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]
    

if __name__ == "__main__":
    
    # -- the collection name tuple
    beer_card = Card("7","diamonds")
    # print(beer_card)
    
    # -- the len()
    deck = frenchDesk()
    # print(len(deck))
    # print(deck[0], deck[-1])
    # Card(rank='2', suit='spades') Card(rank='A', suit='hearts')
    # print(deck[12::13])
    
    # -- loop
    # for item in deck:
    #     print(item)
    # for item in reversed(deck):
    #     print(item)