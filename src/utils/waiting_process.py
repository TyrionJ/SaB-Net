from tqdm import tqdm
from time import sleep


def waiting_proc(r, p):
    remaining = list(range(len(r)))

    with tqdm(desc=None, total=len(r)) as pbar:
        while len(remaining) > 0:
            all_alive = all([j.is_alive() for j in p._pool])
            if not all_alive:
                raise RuntimeError('Some background worker is break.')
            done = [i for i in remaining if r[i].ready()]
            remaining = [i for i in remaining if i not in done]
            pbar.update(len(done))
            sleep(0.02)
