from moleskin import moleskin as M


class Something:
    a = 0


@M.timeit
def fn():
    for i in range(1000000):
        if Something.a:
            Something.a = 1
        else:
            Something.a = 0

M.debug(fn())
