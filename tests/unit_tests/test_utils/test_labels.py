import unittest

import numpy as np

import vak.utils.labels


class TestLabels(unittest.TestCase):
    def test_where(self):
        labels_arr_0 = np.zeros(shape=(10,), dtype=np.int64)
        labels_arr_1 = np.ones(shape=(10,), dtype=np.int64)
        label_arrs = [labels_arr_0, labels_arr_1]
        where_in_labels = vak.utils.labels.where(label_arrs)

        self.assertTrue(list(where_in_labels.keys()) == [0, 1])
        self.assertTrue(where_in_labels[0] == np.asarray(0))
        self.assertTrue(where_in_labels[1] == np.asarray(1))

    def test_where_find_in_arr(self):
        labels_arr_0 = np.zeros(shape=(10,), dtype=np.int64)
        labels_arr_1 = np.ones(shape=(10,), dtype=np.int64)
        labels_arr_2 = np.zeros(shape=(10,), dtype=np.int64)
        labels_arr_3 = np.ones(shape=(10,), dtype=np.int64)
        label_arrs = [labels_arr_0, labels_arr_1, labels_arr_2, labels_arr_3]
        where_in_labels, where_in_arr = vak.utils.labels.where(label_arrs, find_in_arr=True)

        self.assertTrue(list(where_in_labels.keys()) == [0, 1])
        self.assertTrue(
            np.array_equal(where_in_labels[0], np.asarray([0, 2]))
        )
        self.assertTrue(
            np.array_equal(where_in_labels[1], np.asarray([1, 3]))
        )

        self.assertTrue(all([type(v) == dict] for v in where_in_arr.values()))

        self.assertTrue(
            list(where_in_arr[0].keys()) == [0, 2]
        )
        self.assertTrue(
            all(
                [
                    np.array_equal(arr, np.arange(10)) for arr in list(where_in_arr[0].values())
                ]
            )
        )
        self.assertTrue(
            list(where_in_arr[1].keys()) == [1, 3]
        )
        self.assertTrue(
            all(
                [
                    np.array_equal(arr, np.arange(10)) for arr in list(where_in_arr[1].values())
                ]
            )
        )

    def test_sort(self):
        uniq_labels = list('abc')
        occurrences = [
            np.arange(4),
            np.arange(2),
            np.arange(10),
        ]

        uniq_sorted, occur_sorted = vak.utils.labels.sort(uniq_labels, occurrences)
        occur_sorted_expected = [
            np.arange(2),
            np.arange(4),
            np.arange(10),
        ]
        self.assertTrue(uniq_sorted == ['b', 'a', 'c'])
        self.assertTrue(
            all(
                [np.array_equal(arr1, arr2) for arr1, arr2 in zip(occur_sorted, occur_sorted_expected)]
            )
        )

    def test_to_map(self):
        labelset = set(list('abcde'))
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=False)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset)  # because map_unlabeled=False
        )

        labelset = set(list('abcde'))
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=True)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset) + 1  # because map_unlabeled=True
        )

        labelset = {1, 2, 3, 4, 5, 6}
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=False)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset)  # because map_unlabeled=False
        )

        labelset = {1, 2, 3, 4, 5, 6}
        labelmap = vak.utils.labels.to_map(labelset, map_unlabeled=True)
        self.assertTrue(
            type(labelmap) == dict
        )
        self.assertTrue(
            len(labelmap) == len(labelset) + 1  # because map_unlabeled=True
        )

    def test_to_set(self):
        labels1 = [1, 1, 1, 1, 2, 2, 3, 3, 3]
        labels2 = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
        labels_list = [labels1, labels2]
        labelset = vak.utils.labels.to_set(labels_list)
        self.assertTrue(
            type(labelset) == set
        )
        self.assertTrue(
            labelset == {1, 2, 3}
        )


if __name__ == '__main__':
    unittest.main()
