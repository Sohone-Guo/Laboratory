import pymongo

from util_docomo_beijing.nlp.os import read_folder_content

import os

if __name__ == "__main__":
    # define the database
    client = pymongo.MongoClient('192.168.2.130', 27017) 

    db = client.nlp_english
    collection = db.kaggle


    # files
    files = read_folder_content("./2/")

    for idx, file in enumerate(files):
        print(idx)
        input, target = file.split("<label:>")
        data = {"input": input.strip(),
                "target": target.strip()}
        collection.insert_one(data).inserted_id
         
