import datasets

def load_tinystories():
    return datasets.load_dataset("roneneldan/TinyStories", "default")