import os


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


if __name__ == '__main__':
    print(list(map(int, "512,512".split(","))))
