import jax


class PRNGSequence:
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def take_n(self, n):
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        return keys[1:]


class Count:
    def __init__(self, n: int):
        self.count = 0
        self.n = n

    def __call__(self):
        bingo = (self.count + 1) == self.n
        self.count = (self.count + 1) % self.n
        return bingo
