def mkdir_p(my_path):
    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(my_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else:
            raise
