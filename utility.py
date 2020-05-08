import pathlib
import shutil
import random
import os

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

NUM_ELEMS = 10335


def fix_seg_names(seg_path=DATA_PATH / 'seg'):
    segs = [seg for seg in seg_path.glob('*.png') if seg.is_file() and not seg.stem.startswith('.')]
    assert len(segs) == NUM_ELEMS, 'Elementi mancanti! Riscarica il dataset'
    segs = [seg for seg in segs if seg.stem.startswith('seg_')]
    if not segs:
        print('Names already fixed!')
        return
    for seg in segs:
        id_ = seg.stem.split('_')[1]
        seg.rename(seg.parent / ('img_' + id_ + '.png'))


def split_dataset(
        train_val_test_tuple=(.7, .1, .2),
        data_dir=DATA_PATH,
        train_dir=TRAIN_PATH,
        val_dir=VAL_PATH,
        test_dir=TEST_PATH
    ):
    """
    train_val_test nel formato (train%, val%, test%)
    esempio (0.7, 0.15, 0.15)
    """
    t = train_val_test_tuple
    assert sum(t) == 1, 'Percentuale sbagliata! La somma non fa 1'

    count = lambda dir_: len(list((dir_ / 'rgb').glob('*')))
    total_elems = count(data_dir)
    assert total_elems == NUM_ELEMS, 'Mancano elementi nel dataset'

    train_elems = int(t[0] * total_elems)
    val_elems = int(t[1] * total_elems)
    test_elems = int(t[2] * total_elems)

    s = sum([train_elems, val_elems, test_elems])
    if s < NUM_ELEMS:
        diff = NUM_ELEMS - s
        test_elems += diff

    _split_train_val(train_elems, data_dir, train_dir, val_dir)
    _split_train_val(test_elems, val_dir, test_dir, val_dir)

    assert count(train_dir) == train_elems
    assert count(val_dir) == val_elems
    assert count(test_dir) == test_elems


def _split_train_val(num_elem, data_dir=DATA_PATH, train_dir=TRAIN_PATH, val_dir=VAL_PATH):
    rgb_dir = data_dir / 'rgb'
    seg_dir = data_dir / 'seg'

    train_rgb = train_dir / 'rgb'
    train_seg = train_dir / 'seg'

    val_rgb = val_dir / 'rgb'
    val_seg = val_dir / 'seg'

    all_rgb = [f for f in rgb_dir.iterdir() if f.is_file()]
    all_seg = [f for f in seg_dir.iterdir() if f.is_file()]

    selected_rgb = random.sample(all_rgb, num_elem)
    selected_seg = [seg for seg in all_seg if seg.stem in map(lambda path: path.stem, selected_rgb)]

    assert len(selected_rgb) == len(selected_seg), 'Mismatch fra immagine e maschera!'

    val_rgb = list(set(all_rgb) - set(selected_rgb))
    val_seg = list(set(all_seg) - set(selected_seg))

    os.makedirs(train_rgb, exist_ok=True)
    os.makedirs(train_seg, exist_ok=True)
    os.makedirs(val_rgb, exist_ok=True)
    os.makedirs(val_seg, exist_ok=True)

    move(selected_rgb, train_rgb)
    move(selected_seg, train_seg)

    move(val_rgb, val_rgb)
    move(val_seg, val_seg)


def restore_dataset(data_dir=DATA_PATH, train_dir=TRAIN_PATH, val_dir=VAL_PATH, test_dir=TEST_PATH):
    rgb_dir = data_dir / 'rgb'
    seg_dir = data_dir / 'seg'

    train_rgb = [f for f in (train_dir / 'rgb').iterdir() if f.is_file()]
    train_seg = [f for f in (train_dir / 'seg').iterdir() if f.is_file()]

    val_rgb = [f for f in (val_dir / 'rgb').iterdir() if f.is_file()]
    val_seg = [f for f in (val_dir / 'seg').iterdir() if f.is_file()]

    test_rgb = [f for f in (test_dir / 'rgb').iterdir() if f.is_file()]
    test_seg = [f for f in (test_dir / 'seg').iterdir() if f.is_file()]

    if all(not bool(dir_) for dir_ in (train_rgb, test_rgb, val_rgb)):
        print('Restore already done')
        return

    move(train_rgb, rgb_dir)
    move(val_rgb, rgb_dir)
    move(test_rgb, rgb_dir)

    move(train_seg, seg_dir)
    move(val_seg, seg_dir)
    move(test_seg, seg_dir)

    print('Restore eseguito!')


def move(data, dir_):
    for d in data:
        try:
            shutil.move(str(d), str(dir_))
        except shutil.Error:  # strano bug che esegue di nuovo la move dopo aver terminato
            return
