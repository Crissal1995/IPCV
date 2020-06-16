import pathlib
import shutil
import random
import os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = False

"""
Bisogna avere il dataset originale in una cartella chiamata "sunrgb"
nella root del progetto, dopodiché in "dataset" verranno create le tre cartelle di split,
ovvero train, val e test.


Di default la percentuale è 70% train, 10% validation, 20% test, ma è possibile modificarla
passando una tupla alla funzione split_dataset.
La somma delle percentuali deve ovviamente fare 1.

ex split_dataset( (.7, .2, .1) )

root
  |- sunrgb
        | - mat [opzionale]
        | - rgb
        | - seg
"""

DATA_PATH = pathlib.Path() / 'sunrgb'
DATASET_PATH = pathlib.Path() / 'dataset'

TRAIN_PATH = DATASET_PATH / 'train'
VAL_PATH = DATASET_PATH / 'val'
TEST_PATH = DATASET_PATH / 'test'

NUM_ELEMENTS = 10335
SEED = 0
random.seed(SEED)


def change_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)


def fix_seg_names(seg_path=DATA_PATH / 'seg'):
    segs = [seg for seg in seg_path.glob('*.png') if seg.is_file() and not seg.stem.startswith('.')]
    assert len(segs) == NUM_ELEMENTS, 'fix_seg_names() già eseguita oppure dataset incompleto'
    segs = [seg for seg in segs if seg.stem.startswith('seg_')]
    if not segs:
        print('Nomi già fixati')
        return

    t = tqdm(total=len(segs)) if tqdm else None

    for seg in segs:
        id_ = seg.stem.split('_')[1]
        seg.rename(seg.parent / ('img_' + id_ + '.png'))
        if tqdm:
            t.update()

    if tqdm:
        t.close()


def split_dataset(
        train_val_test_tuple=(.7, .1, .2),
        data_dir=DATA_PATH,
        train_dir=TRAIN_PATH,
        val_dir=VAL_PATH,
        test_dir=TEST_PATH,
        is_kaggle=False
):
    """
    train_val_test nel formato (train%, val%, test%)
    esempio (0.7, 0.15, 0.15)

    se verranno passati due soli valori, il validation sarà vuoto
    e sarà tutto train e test
    """
    t = train_val_test_tuple
    assert sum(t) == 1, 'Percentuale sbagliata! La somma non fa 1'

    assert len(t) in (2, 3), \
        'Numero di parametri nella tupla errato! Solo due (train, test) o tre (train, val, test) permessi'

    if len(t) == 2:
        t = (t[0], 0, t[1])

    count = lambda dir_: len(list((dir_ / 'rgb').glob('*')))

    total_elems = count(data_dir)
    assert total_elems == NUM_ELEMENTS, 'Mancano elementi nel dataset. Forse è già stato splittato?'

    train_elems = int(t[0] * total_elems)
    val_elems = int(t[1] * total_elems)
    test_elems = int(t[2] * total_elems)

    s = sum([train_elems, val_elems, test_elems])
    diff = NUM_ELEMENTS - s
    if diff:
        test_elems += diff

    t = tqdm(total=2*NUM_ELEMENTS) if tqdm else None

    _move_from_folder(train_elems, data_dir, train_dir, t, is_kaggle)
    _move_from_folder(val_elems, data_dir, val_dir, t, is_kaggle)
    _move_from_folder(test_elems, data_dir, test_dir, t, is_kaggle)

    if tqdm:
        t.close()

    assert count(train_dir) == train_elems
    assert count(val_dir) == val_elems
    assert count(test_dir) == test_elems


def _move_from_folder(num_elem, from_dir, to_dir, tqdm_progress=None, is_kaggle=False):
    from_dir_rgb = from_dir / 'rgb'
    from_dir_seg = from_dir / 'seg'

    to_dir_rgb = to_dir / 'rgb'
    to_dir_seg = to_dir / 'seg'

    os.makedirs(to_dir_rgb, exist_ok=True)
    os.makedirs(to_dir_seg, exist_ok=True)

    if not num_elem:
        return

    assert all(dir_.exists() for dir_ in (from_dir_rgb, from_dir_seg)), \
        'Cartelle non valide! Al loro interno devono contenere necessariamente "rgb" e "seg"'

    rgbs = [f for f in from_dir_rgb.glob('*.jpg') if _is_valid_file(f)]
    segs = [f for f in from_dir_seg.glob('*.png') if _is_valid_file(f)]

    assert len(rgbs) == len(segs), 'Mismatch in numero fra immagini e maschere!'

    selected_rgb = random.sample(rgbs, num_elem)
    selected_stems = [rgb.stem for rgb in selected_rgb]
    selected_seg = [seg for seg in segs if seg.stem in selected_stems]

    assert len(selected_rgb) == len(selected_seg), 'Mismatch di elementi fra immagini e maschere!'

    if not is_kaggle:
        move(selected_rgb, to_dir_rgb, tqdm_progress)
        move(selected_seg, to_dir_seg, tqdm_progress)
    else:
        copy(selected_rgb, to_dir_rgb, tqdm_progress)
        copy(selected_seg, to_dir_seg, tqdm_progress)


def restore_dataset(data_dir=DATA_PATH, train_dir=TRAIN_PATH, val_dir=VAL_PATH, test_dir=TEST_PATH):
    rgb_dir = data_dir / 'rgb'
    seg_dir = data_dir / 'seg'

    train_rgb = [f for f in (train_dir / 'rgb').iterdir() if _is_valid_file(f)]
    train_seg = [f for f in (train_dir / 'seg').iterdir() if _is_valid_file(f)]

    val_rgb = [f for f in (val_dir / 'rgb').iterdir() if _is_valid_file(f)]
    val_seg = [f for f in (val_dir / 'seg').iterdir() if _is_valid_file(f)]

    test_rgb = [f for f in (test_dir / 'rgb').iterdir() if _is_valid_file(f)]
    test_seg = [f for f in (test_dir / 'seg').iterdir() if _is_valid_file(f)]

    seg_dirs = (train_seg, val_seg, test_seg)
    rgb_dirs = (train_rgb, val_rgb, test_rgb)
    dirs = seg_dirs + rgb_dirs

    assert all(len(rgb) == len(seg) for (rgb, seg) in zip(rgb_dirs, seg_dirs)), \
        'Mismatch in numero fra RGB e SEG in almeno una cartella'

    if all(not d for d in rgb_dirs):
        print('Restore già fatto')
        return

    assert sum(len(d) for d in dirs) == 2*NUM_ELEMENTS, 'Mancano elementi nelle cartelle! Riscaricare il dataset'

    t = tqdm(total=2*NUM_ELEMENTS) if tqdm else None

    move(train_rgb, rgb_dir, t)
    move(val_rgb, rgb_dir, t)
    move(test_rgb, rgb_dir, t)

    move(train_seg, seg_dir, t)
    move(val_seg, seg_dir, t)
    move(test_seg, seg_dir, t)

    if tqdm:
        t.close()


def move(data, dir_, tqdm_progress=None):
    for d in data:
        shutil.move(str(d), str(dir_))
        if tqdm_progress:
            tqdm_progress.update()


def copy(data, dir_, tqdm_progress=None):
    for d in data:
        shutil.copy(str(d), str(dir_))
        if tqdm_progress:
            tqdm_progress.update()


def _is_valid_file(file):
    return file.is_file() and not file.stem.startswith('.')


def info():
    def _walk(path):
        print('Inside', path, ':')
        for root, dirs, files in os.walk(path):
            if root.startswith('.'):
                continue
            if not files:
                continue
            print(' | DIR {} : FILES {}'.format(root, len(files)))

    _walk(DATA_PATH)
    _walk(DATASET_PATH)

