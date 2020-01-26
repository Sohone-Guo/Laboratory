import re  

def position_collection(key:str, value:str):
    """ Return the position of key 
    in the value.
    """
    def position_finder(k_):
        return [i.start() for i in re.finditer(k_, value)]
        
    key_, idx, collection = ("", 0, [])
    
    while idx < len(key):
        key_ += key[idx]
        position_ =  position_finder(key_)
        idx += 1
        
        if len(position_) == 0 and len(key_) == 1:
            collection.append((key_, idx-1, -1))
            key_ = ""
        elif len(position_) == 0 and len(key_) > 1:
            idx_start = idx - len(key_)
            key_ = key_[:-1]
            collection.append((key_, idx_start, position_finder(key_)[0]))
            idx -= 1
            key_ = ""
    else:
        if len(key_) == 0:
            pass
        elif len(key_) == 1:
            collection.append((key_, idx-1, -1))
        else:
            idx_start = idx - len(key_)
            collection.append((key_, idx_start, position_[0]))
    return collection