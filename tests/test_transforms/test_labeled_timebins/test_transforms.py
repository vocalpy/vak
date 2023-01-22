import numpy as np
import pytest

import vak


from .test_functional import (
    FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
    MAX_ABS_DIFF,
    SPECT_FILES_TO_SKIP,
    TIMEBIN_DUR_FOR_PARAMETRIZE,
    POSTPROCESS_PARAMS_ARGVALS,
)


class TestFromSegments:
    def test_init(self):
        from_segments_tfm = vak.transforms.labeled_timebins.FromSegments()
        assert isinstance(from_segments_tfm, vak.transforms.labeled_timebins.FromSegments)

    @pytest.mark.parametrize(
        'annot, spect_path, labelset',
        FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
    )
    def test_call(self, annot, spect_path, labelset):
        labelset = vak.converters.labelset_to_set(labelset)
        labelmap = vak.labels.to_map(labelset, True)

        spect_dict = vak.files.spect.load(spect_path)
        timebins = spect_dict['t']

        try:
            lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
        except KeyError:
            pytest.skip(
                'Annotation with label not in labelset, would not include in dataset'
            )

        from_segments_tfm = vak.transforms.labeled_timebins.FromSegments(unlabeled_label=labelmap['unlabeled'])
        lbl_tb = from_segments_tfm(
            lbls_int,
            annot.seq.onsets_s,
            annot.seq.offsets_s,
            timebins,
        )
        assert lbl_tb.shape == timebins.shape
        assert all(
            [lbl in lbls_int for lbl in np.unique(lbls_int)]
        )


class TestToLabels:
    @pytest.mark.parametrize(
        'labelset',
        [tup[2] for tup in FROM_SEGMENTS_PARAMETRIZE_ARGVALS],
    )
    def test_init(self, labelset):
        # Note that we add an 'unlabeled' class because post-processing transforms *require* it
        # This is default, just making it explicit
        labelset = vak.converters.labelset_to_set(labelset)
        labelmap = vak.labels.to_map(labelset, map_unlabeled=True)

        to_labels_tfm = vak.transforms.labeled_timebins.ToLabels(
            labelmap=labelmap,
        )
        assert isinstance(to_labels_tfm, vak.transforms.labeled_timebins.ToLabels)

    @pytest.mark.parametrize(
        "lbl_tb, labelmap, labels_expected_int",
        [
            (np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0]), {'unlabeled': 0, 'a': 1, 'b': 2}, [1, 2]),
            (np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 0]), {'unlabeled': 0, '1': 1, '2': 2}, [1, 2]),
            (np.array([0, 0, 21, 21, 0, 0, 22, 22, 0, 0]), {'unlabeled': 0, '21': 21, '22': 22}, [21, 22]),
            (np.array([0, 0, 11, 11, 0, 0, 12, 12, 0, 0]), {'unlabeled': 0, '11': 11, '12': 12}, [11, 12]),
        ]
    )
    def test_call(self, lbl_tb, labelmap, labels_expected_int):
        # Note that we add an 'unlabeled' class because post-processing transforms *require* it
        # This is default, just making it explicit
        labelmap = vak.labels.multi_char_labels_to_single_char(
            labelmap, skip=('unlabeled',)
        )
        labelmap_inv = {v: k for k, v in labelmap.items()}
        labels_expected = ''.join([labelmap_inv[lbl_int] for lbl_int in labels_expected_int])

        to_labels_tfm = vak.transforms.labeled_timebins.ToLabels(
            labelmap=labelmap,
        )
        labels = to_labels_tfm(lbl_tb)
        assert labels == labels_expected

    @pytest.mark.parametrize(
        'annot, spect_path, labelset',
        FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
    )
    def test_call_real_data(
            self, annot, spect_path, labelset
    ):
        """test that ``to_labels_with_postprocessing`` recovers labels from real data"""
        labelset = vak.converters.labelset_to_set(labelset)
        labelmap = vak.labels.to_map(labelset)
        # next line, convert all labels to single characters
        # we can easily compare strings we get back with expected;
        # this is what core.eval does
        labelmap = vak.labels.multi_char_labels_to_single_char(
            labelmap, skip=('unlabeled',)
        )
        TIMEBINS_KEY = "t"

        if any(
            str(spect_path).endswith(spect_file_to_skip)
            for spect_file_to_skip in SPECT_FILES_TO_SKIP
        ):
            pytest.skip(
                "Can't round trip segments -> lbl_tb -> segments "
                "because of small silent gap durations + large time bin durations"
            )

        try:
            lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
        except KeyError:
            pytest.skip(
                'Annotation with label not in labelset, would not include in dataset'
            )

        timebins = vak.files.spect.load(spect_path)[TIMEBINS_KEY]

        lbl_tb = vak.transforms.labeled_timebins.from_segments(
            lbls_int,
            annot.seq.onsets_s,
            annot.seq.offsets_s,
            timebins,
            unlabeled_label=labelmap["unlabeled"],
        )

        to_labels_tfm = vak.transforms.labeled_timebins.ToLabels(
            labelmap=labelmap,
        )
        labels = to_labels_tfm(lbl_tb)

        labelmap_multi_inv = {v: k for k, v in
                              labelmap.items()}
        labels_expected = "".join(
            [labelmap_multi_inv[lbl_int] for lbl_int in lbls_int]
        )
        assert labels == labels_expected


class TestToSegments:
    @pytest.mark.parametrize(
        'labelset',
        [tup[2] for tup in FROM_SEGMENTS_PARAMETRIZE_ARGVALS],
    )
    def test_init(self, labelset):
        # Note that we add an 'unlabeled' class because post-processing transforms *require* it
        # This is default, just making it explicit
        labelset = vak.converters.labelset_to_set(labelset)
        labelmap = vak.labels.to_map(labelset, map_unlabeled=True)

        to_segments_tfm = vak.transforms.labeled_timebins.ToSegments(
            labelmap=labelmap,
        )
        assert isinstance(to_segments_tfm, vak.transforms.labeled_timebins.ToSegments)

    @pytest.mark.parametrize(
        'annot, spect_path, labelset',
        FROM_SEGMENTS_PARAMETRIZE_ARGVALS,
    )
    def test_call_real_data(self, annot, spect_path, labelset):
        labelset = vak.converters.labelset_to_set(labelset)
        labelmap = vak.labels.to_map(labelset)

        TIMEBINS_KEY = "t"

        if any(
                str(spect_path).endswith(spect_file_to_skip)
                for spect_file_to_skip in SPECT_FILES_TO_SKIP
        ):
            pytest.skip(
                "Can't round trip segments -> lbl_tb -> segments "
                "because of small silent gap durations + large time bin durations"
            )

        try:
            lbls_int = [labelmap[lbl] for lbl in annot.seq.labels]
        except KeyError:
            pytest.skip(
                'Annotation with label not in labelset, would not include in dataset'
            )

        timebins = vak.files.spect.load(spect_path)[TIMEBINS_KEY]

        lbl_tb = vak.transforms.labeled_timebins.from_segments(
            lbls_int,
            annot.seq.onsets_s,
            annot.seq.offsets_s,
            timebins,
            unlabeled_label=labelmap["unlabeled"],
        )

        to_segments_tfm = vak.transforms.labeled_timebins.ToSegments(
            labelmap=labelmap,
        )

        labels, onsets_s, offsets_s = to_segments_tfm(
            lbl_tb, timebins
        )

        assert np.all(np.char.equal(labels, annot.seq.labels))
        assert np.all(np.abs(annot.seq.onsets_s - onsets_s) < MAX_ABS_DIFF)
        assert np.all(np.abs(annot.seq.offsets_s - offsets_s) < MAX_ABS_DIFF)


class TestPostprocess:
    @pytest.mark.parametrize(
        'min_segment_dur, majority_vote, timebin_dur',
        # keep just the argvals we need to instantiate
        [argvals[3:5] + (TIMEBIN_DUR_FOR_PARAMETRIZE,) for argvals in POSTPROCESS_PARAMS_ARGVALS]
    )
    def test_init(self, min_segment_dur, majority_vote, timebin_dur):
        # Note that we add an 'unlabeled' class
        # because post-processing transforms *require* it
        # This is default, just making it explicit
        to_labels_tfm = vak.transforms.labeled_timebins.PostProcess(
            min_segment_dur=min_segment_dur,
            majority_vote=majority_vote,
            timebin_dur=timebin_dur,
        )
        assert isinstance(to_labels_tfm, vak.transforms.labeled_timebins.PostProcess)

    @pytest.mark.parametrize(
        'lbl_tb, timebin_dur, unlabeled_label, min_segment_dur, majority_vote, lbl_tb_expected',
        POSTPROCESS_PARAMS_ARGVALS
    )
    def test_call(self, lbl_tb, timebin_dur, unlabeled_label, min_segment_dur, majority_vote, lbl_tb_expected):
        # Note that we add an 'unlabeled' class because post-processing transforms *require* it
        # This is default, just making it explicit
        postprocess_tfm = vak.transforms.labeled_timebins.PostProcess(
            min_segment_dur=min_segment_dur,
            majority_vote=majority_vote,
            timebin_dur=timebin_dur,
        )

        lbl_tb = postprocess_tfm(
            lbl_tb
        )

        assert np.all(np.equal(lbl_tb, lbl_tb_expected))

