fileHead = '/filehome/PythonProjects/learn_ai_python/image/'


def image_in(filename='', fileHead=fileHead):
    Path = fileHead + 'p1.png'
    if filename == 'p1':
        Path = fileHead + 'p1.png'
    elif filename == 'lenna':
        Path = fileHead + 'lenna.png'
    elif filename == 'damaged':
        Path = fileHead+'damaged.png'
    elif filename == 'repair':
        Path = fileHead+'repair.png'
    return Path
