import vak.files.spect
import vak.labels


def test_to_map():
    labelset = set(list("abcde"))
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset)  # because map_unlabeled=False

    labelset = set(list("abcde"))
    labelmap = vak.labels.to_map(labelset, map_unlabeled=True)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset) + 1  # because map_unlabeled=True

    labelset = {1, 2, 3, 4, 5, 6}
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset)  # because map_unlabeled=False

    labelset = {1, 2, 3, 4, 5, 6}
    labelmap = vak.labels.to_map(labelset, map_unlabeled=True)
    assert type(labelmap) == dict
    assert len(labelmap) == len(labelset) + 1  # because map_unlabeled=True


def test_to_set():
    labels1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    labels2 = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
    labels_list = [labels1, labels2]
    labelset = vak.labels.to_set(labels_list)
    assert type(labelset) == set
    assert labelset == {1, 2, 3}
