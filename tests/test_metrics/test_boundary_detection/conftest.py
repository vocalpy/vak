"""
Pytest conftest module, for test_boundary_detection_module.

At the time of adding the test_boundary_detection_module, 
this does not contain any fixtures or test configuration per se;
it is just a place to put the really long list of unit test cases 
that parametrize the unit tests,
so that those modules are a little more readable.
"""
import attr
import torch


@attr.define
class FindHitsTestCase:
    """Class representing a test case for the
    :func:`test_find_hits` unit test"""
    target: torch.FloatTensor = attr.field(converter=torch.FloatTensor)
    preds: torch.FloatTensor = attr.field(converter=torch.FloatTensor)
    expected_hits_target: torch.LongTensor = attr.field(converter=torch.LongTensor)
    expected_hits_preds: torch.LongTensor = attr.field(converter=torch.LongTensor)
    expected_diffs: torch.FloatTensor = attr.field(converter=torch.FloatTensor)
    tolerance: float | None | str = attr.field(default=None)
    decimals: bool | int | None  = attr.field(default=None)
    expected_num_hits: torch.LongTensor = attr.field(init=False)

    def __attrs_post_init__(self):
        self.expected_num_hits = torch.tensor(
            self.expected_hits_target.numel()
        ).long()


# for `find_hits` all test cases are 1-D tensors
FIND_HITS_TEST_CASES = [
    # # 1-D
    # ## default tolerance and precision
    # ### all hit
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.000, 5.000, 10.000, 15.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0, 0, 0, 0],
    ),
    # ### no hits
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.000, 6.000, 11.000, 16.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
    ),
    # ### no > hits > all
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.000, 6.000, 10.000, 16.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0],
    ),
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.000, 5.000, 10.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
    ),
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.000, 5.000, 10.000, 15.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
    ),
    # ## tolerance of 0.5
    # ### all hits
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.500, 5.500, 10.500, 15.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5, 0.5],
    ),
    # ### no hits
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.500, 6.500, 11.500, 16.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
    ),
    # ### no > hits > all
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.500, 6.500, 10.500, 16.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0.5],
    ),
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.500, 5.500, 10.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
    ),
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.500, 5.500, 10.500, 15.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
    ),
    # ### multiple hits, tests we only keep one
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.500, 1.500, 5.500, 10.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5],
    ),
    FindHitsTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 3, 6],
        expected_diffs=[0.25, 0, 0.5],
    ),
    # ## default tolerance, precision=3 (happens to be default)
    # ### all hits
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[0.0004, 5.0004, 10.0004, 15.0004],
        tolerance=None,
        decimals=3,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0., 0., 0., 0.],
    ),
    # ### no hits
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[1.0001, 6.0001, 11.0001, 16.0001],
        tolerance=None,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
    ),
    # ### no > hits > all
    FindHitsTestCase(
        target=[1.0001, 6.0001, 10.0004, 16.0001],
        preds=[0.0001, 5.0001, 10.0001, 15.0001],
        tolerance=None,
        decimals=3,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0],
    ),
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[0.0004, 5.0004, 10.0004],
        tolerance=None,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
    ),
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.0004, 5.0004, 10.0004, 15.0004],
        tolerance=None,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
    ),
    # ## tolerance of 0.5, decimals=3 (default)
    # ### all hits
    FindHitsTestCase(
        target=[0.0004, 5.0004, 10.0004, 15.0004],
        preds=[0.5001, 5.5001, 10.5001, 15.5001],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5, 0.5],
    ),
    # ### no hits
    FindHitsTestCase(
        target=[0.0004, 5.0004, 10.0004, 15.0004],
        preds=[1.5001, 6.5001, 11.5001, 16.5001],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
    ),
    # ### no > hits > all
    FindHitsTestCase(
        target=[0.0004, 5.0004, 10.0004, 15.0004],
        preds=[1.5001, 6.5001, 10.5001, 16.5001],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0.5],
    ),
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[0.5004, 5.5004, 10.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
    ),
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.5004, 5.5004, 10.5004, 15.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
    ),
    # ### multiple hits, tests we only keep one
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.5004, 1.5004, 5.5004, 10.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5],
    ),
    FindHitsTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.2504, 0.5004, 2.5004, 5.0004, 5.5004, 7.5004, 10.5004, 11.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 3, 6],
        expected_diffs=[0.25, 0, 0.5],
    ),
    # # edge cases
    # no boundaries in target
    FindHitsTestCase(
        target=[],
        preds=[1.0, 2.0, 3.0],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=torch.FloatTensor([]),
    ),
    # no boundaries in preds
    FindHitsTestCase(
        target=[1.0, 2.0, 3.0],
        preds=[],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=torch.FloatTensor([]),
    ),
    # only one boundary in ref/hyp
    FindHitsTestCase(
        target=[1.0],
        preds=[1.0],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0],
        expected_hits_preds=[0],
        expected_diffs=torch.FloatTensor([0]),
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/119
    FindHitsTestCase(
        target=[2.244, 2.262],
        preds=[2.254],
        tolerance=0.01,
        decimals=3,
        expected_hits_target=[1],
        expected_hits_preds=[0],
        expected_diffs=[0.008],
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/170
    FindHitsTestCase(
        target=[],
        preds=[],
        tolerance=None,
        decimals=None,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
    ),
]


def convert_expected_hits(val):
    """Convert expected hits_(target/preds) to a list of torch.LongTensor,
    so we can index into list inside unit test
    """
    if all(
        [isinstance(element, int) for element in val]
    ):
        # batch size == 1
        return [torch.LongTensor(val)]
    elif all(
        [isinstance(element, list) for element in val]
    ):
        # batch size > 1
        return [
            torch.LongTensor(element) for element in val
        ]
    else:
        raise ValueError(
            f"`convert_expected_hits` got unexpected value, {val}, of type {type(val)}"
        )


def convert_expected_diffs(val):
    """Convert expected diffs to a list of torch.FloatTensor,
    so we can index into list inside unit test
    """
    if all(
        [isinstance(element, (float, int)) for element in val]
    ):
        # batch size == 1
        return [torch.FloatTensor(val)]
    elif all(
        [isinstance(element, list) for element in val]
    ):
        # batch size > 1
        return [
            torch.FloatTensor(element) for element in val
        ]
    else:
        raise ValueError(
            f"`convert_expected_diffs` got unexpected value, {val}, of type {type(val)}"
        )



def convert_ir_metric(val: float | int | list[float] | list[int]) -> torch.FloatTensor:
    """Convert a float or int value to a zero-dimensional torch.FloatTensor"""
    if isinstance(val, (float, int)):
        return torch.tensor(val).float()
    elif isinstance(val, list):
        return torch.FloatTensor(val)
    else:
        raise ValueError(
            f"`convert_ir_metric` got unexpected value, {val}, of type {type(val)}"
        )


@attr.define
class PRFRTestCase:
    """Class representing a test case for the unit test 
    :func:`test_precision_recall_fscore_rval`
    """
    target: torch.FloatTensor = attr.field(converter=torch.FloatTensor)
    preds: torch.FloatTensor = attr.field(converter=torch.FloatTensor)
    expected_hits_target: list[torch.LongTensor] = attr.field(converter=convert_expected_hits)
    expected_hits_preds: list[torch.LongTensor] = attr.field(converter=convert_expected_hits)
    expected_diffs: list[torch.FloatTensor] = attr.field(converter=convert_expected_diffs)
    expected_precision: torch.FloatTensor = attr.field(converter=convert_ir_metric)
    expected_recall: torch.FloatTensor = attr.field(converter=convert_ir_metric)
    expected_fscore: torch.FloatTensor = attr.field(converter=convert_ir_metric)
    expected_rval: torch.FloatTensor = attr.field(converter=convert_ir_metric)
    tolerance: float | None | str = attr.field(default=None)
    decimals: bool | int | None  = attr.field(default=None)
    ignore_val: int | None = attr.field(default=None)
    expected_num_hits: int = attr.field(init=False)

    def __attrs_post_init__(self):
        self.expected_num_hits = [
            torch.tensor(expected_hits.numel()).long()
            for expected_hits in self.expected_hits_target
        ]


PRECISION_RECALL_FSCORE_RVAL_TEST_CASES = [
    # # 1-D
    # ## default tolerance and precision
    # ### all hit
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.000, 5.000, 10.000, 15.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0, 0, 0, 0],
        expected_precision=[1.0],
        expected_recall=[1.0],
        expected_fscore=[1.0],
        expected_rval=[0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.000, 6.000, 11.000, 16.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[0.0],
        expected_recall=[0.0],
        expected_fscore=[0.0],
        expected_rval=[0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.000, 6.000, 10.000, 16.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0],
        expected_precision=[0.25],
        expected_recall=[0.25],
        expected_fscore=[0.25],
        expected_rval=[0.359],
    ),
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.000, 5.000, 10.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
        expected_precision=[1.0],
        expected_recall=[0.75],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.823],
    ),
    PRFRTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.000, 5.000, 10.000, 15.000],
        tolerance=None,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
        expected_precision=[0.75],
        expected_recall=[1.0],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.715],
    ),
    # ## tolerance of 0.5
    # ### all hits
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.500, 5.500, 10.500, 15.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5, 0.5],
        expected_precision=[1.0],
        expected_recall=[1.0],
        expected_fscore=[1.0],
        expected_rval=[0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.500, 6.500, 11.500, 16.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[0.0],
        expected_recall=[0.0],
        expected_fscore=[0.0],
        expected_rval=[0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[1.500, 6.500, 10.500, 16.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0.5],
        expected_precision=[0.25],
        expected_recall=[0.25],
        expected_fscore=[0.25],
        expected_rval=[0.359],
    ),
    PRFRTestCase(
        target=[0.000, 5.000, 10.000, 15.000],
        preds=[0.500, 5.500, 10.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
        expected_precision=[1.0],
        expected_recall=[0.75],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.823],
    ),
    PRFRTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.500, 5.500, 10.500, 15.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
        expected_precision=[0.75],
        expected_recall=[1.0],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.715],
    ),
    # ### multiple hits, tests we only keep one
    PRFRTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.500, 1.500, 5.500, 10.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5],
        expected_precision=[0.75],
        expected_recall=[1.0],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.715],
    ),
    PRFRTestCase(
        target=[0.000, 5.000, 10.000],
        preds=[0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 3, 6],
        expected_diffs=[[0.25, 0, 0.5]],
        expected_precision=[0.375],
        expected_recall=[1.0],
        expected_fscore=[0.5454545454545454],
        expected_rval=[-0.422],
    ),
    # ## default tolerance, precision=3 (happens to be default)
    # ### all hits
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[0.0004, 5.0004, 10.0004, 15.0004],
        tolerance=None,
        decimals=3,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0., 0., 0., 0.],
        expected_precision=[1.0],
        expected_recall=[1.0],
        expected_fscore=[1.0],
        expected_rval=[0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[1.0001, 6.0001, 11.0001, 16.0001],
        tolerance=None,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[0.0],
        expected_recall=[0.0],
        expected_fscore=[0.0],
        expected_rval=[0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[1.0001, 6.0001, 10.0004, 16.0001],
        preds=[0.0001, 5.0001, 10.0001, 15.0001],
        tolerance=None,
        decimals=3,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0],
        expected_precision=[0.25],
        expected_recall=[0.25],
        expected_fscore=[0.25],
        expected_rval=[0.359],
    ),
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[0.0004, 5.0004, 10.0004],
        tolerance=None,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
        expected_precision=[1.0],
        expected_recall=[0.75],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.823],
    ),
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.0004, 5.0004, 10.0004, 15.0004],
        tolerance=None,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0, 0, 0],
        expected_precision=[0.75],
        expected_recall=[1.0],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.715],
    ),
    # ## tolerance of 0.5, decimals=3 (default)
    # ### all hits
    PRFRTestCase(
        target=[0.0004, 5.0004, 10.0004, 15.0004],
        preds=[0.5001, 5.5001, 10.5001, 15.5001],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2, 3],
        expected_hits_preds=[0, 1, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5, 0.5],
        expected_precision=[1.0],
        expected_recall=[1.0],
        expected_fscore=[1.0],
        expected_rval=[0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[0.0004, 5.0004, 10.0004, 15.0004],
        preds=[1.5001, 6.5001, 11.5001, 16.5001],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[0.0],
        expected_recall=[0.0],
        expected_fscore=[0.0],
        expected_rval=[0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[0.0004, 5.0004, 10.0004, 15.0004],
        preds=[1.5001, 6.5001, 10.5001, 16.5001],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[2],
        expected_hits_preds=[2],
        expected_diffs=[0.5],
        expected_precision=[0.25],
        expected_recall=[0.25],
        expected_fscore=[0.25],
        expected_rval=[0.359],
    ),
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001, 15.0001],
        preds=[0.5004, 5.5004, 10.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
        expected_precision=[1.0],
        expected_recall=[0.75],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.823],
    ),
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.5004, 5.5004, 10.5004, 15.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 1, 2],
        expected_diffs=[0.5, 0.5, 0.5],
        expected_precision=[0.75],
        expected_recall=[1.0],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.715],
    ),
    # ### multiple hits, tests we only keep one
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.5004, 1.5004, 5.5004, 10.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 2, 3],
        expected_diffs=[0.5, 0.5, 0.5],
        expected_precision=[0.75],
        expected_recall=[1.0],
        expected_fscore=[0.8571428571428571],
        expected_rval=[0.715],
    ),
    PRFRTestCase(
        target=[0.0001, 5.0001, 10.0001],
        preds=[0.2504, 0.5004, 2.5004, 5.0004, 5.5004, 7.5004, 10.5004, 11.5004],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0, 1, 2],
        expected_hits_preds=[0, 3, 6],
        expected_diffs=[[0.25, 0, 0.5]],
        expected_precision=[0.375],
        expected_recall=[1.0],
        expected_fscore=[0.5454545454545454],
        expected_rval=[-0.422],
    ),
    # # edge cases
    # no boundaries in target
    PRFRTestCase(
        target=[],
        preds=[1.0, 2.0, 3.0],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[0.0],
        expected_recall=[0.0],
        expected_fscore=[0.0],
        expected_rval=[0.0],
    ),
    # no boundaries in preds
    PRFRTestCase(
        target=[1.0, 2.0, 3.0],
        preds=[],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[0.0],
        expected_recall=[0.0],
        expected_fscore=[0.0],
        expected_rval=[0.0],
    ),
    # only one boundary in ref/hyp
    PRFRTestCase(
        target=[1.0],
        preds=[1.0],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[0],
        expected_hits_preds=[0],
        expected_diffs=[0],
        expected_precision=[1.0],
        expected_recall=[1.0],
        expected_fscore=[1.0],
        expected_rval=[0.999],
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/119
    PRFRTestCase(
        target=[2.244, 2.262],
        preds=[2.254],
        tolerance=0.01,
        decimals=3,
        expected_hits_target=[1],
        expected_hits_preds=[0],
        expected_diffs=[0.008],
        expected_precision=[1.0],
        expected_recall=[0.5],
        expected_fscore=[(2 * 1.0 * 0.5) / (1 + 0.5)],  # 0.6666666666666666 (repeating)
        expected_rval=[0.646],
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/170
    PRFRTestCase(
        target=[],
        preds=[],
        tolerance=None,
        decimals=None,
        expected_hits_target=[],
        expected_hits_preds=[],
        expected_diffs=[],
        expected_precision=[1.0],
        expected_recall=[1.0],
        expected_fscore=[1.0],
        expected_rval=[1.0],
    ),
    # # 2-D
    # ## default tolerance and precision
    # ### all hits
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        tolerance=None,
        decimals=None,
        expected_hits_target=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_hits_preds=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_diffs=[[0, 0, 0, 0], [0, 0, 0, 0]],
        expected_precision=[1.0, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[1.0, 1.0],
        expected_rval=[0.999, 0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[1.000, 6.000, 11.000, 16.000], [1.000, 6.000, 11.000, 16.000]],
        tolerance=None,
        decimals=None,
        expected_hits_target=[[],[]],
        expected_hits_preds=[[],[]],
        expected_diffs=[[],[]],
        expected_precision=[0.0, 0.0],
        expected_recall=[0.0, 0.0],
        expected_fscore=[0.0, 0.0],
        expected_rval=[0.292, 0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[1.000, 6.000, 10.000, 16.000], [1.000, 6.000, 10.000, 16.000]],
        tolerance=None,
        decimals=None,
        expected_hits_target=[[2],[2]],
        expected_hits_preds=[[2],[2]],
        expected_diffs=[[0],[0]],
        expected_precision=[0.25, 0.25],
        expected_recall=[0.25, 0.25],
        expected_fscore=[0.25, 0.25],
        expected_rval=[0.359, 0.359],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        tolerance=None,
        decimals=None,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0, 0, 0], [0, 0, 0]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.75, 0.75],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.823, 0.823],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        preds=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        tolerance=None,
        decimals=None,
        expected_hits_target=[[0, 1, 2],[0, 1, 2]],
        expected_hits_preds=[[0, 1, 2],[0, 1, 2]],
        expected_diffs=[[0, 0, 0],[0, 0, 0]],
        expected_precision=[0.75, 0.75],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.715, 0.715],
    ),
    # ## tolerance of 0.5
    # ### all hits
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[0.500, 5.500, 10.500, 15.500], [0.500, 5.500, 10.500, 15.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_hits_preds=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_diffs=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        expected_precision=[1.0, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[1.0, 1.0],
        expected_rval=[0.999, 0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[1.500, 6.500, 11.500, 16.500], [1.500, 6.500, 11.500, 16.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[], []],
        expected_hits_preds=[[], []],
        expected_diffs=[[], []],
        expected_precision=[0.0, 0.0],
        expected_recall=[0.0, 0.0],
        expected_fscore=[0.0, 0.0],
        expected_rval=[0.292, 0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[1.500, 6.500, 10.500, 16.500], [1.500, 6.500, 10.500, 16.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[2], [2]],
        expected_hits_preds=[[2], [2]],
        expected_diffs=[[0.5], [0.5]],
        expected_precision=[0.25, 0.25],
        expected_recall=[0.25, 0.25],
        expected_fscore=[0.25, 0.25],
        expected_rval=[0.359, 0.359],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, 15.000]],
        preds=[[0.500, 5.500, 10.500], [0.500, 5.500, 10.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.75, 0.75],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.823, 0.823],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        preds=[[0.500, 5.500, 10.500, 15.500], [0.500, 5.500, 10.500, 15.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[0.75, 0.75],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.715, 0.715],
    ),
    # ### multiple hits, tests we only keep one
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        preds=[[0.500, 1.500, 5.500, 10.500], [0.500, 1.500, 5.500, 10.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 2, 3], [0, 2, 3]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[0.75, 0.75],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.715, 0.715],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        preds=[[0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500], [0.250, 0.500, 2.500, 5.000, 5.500, 7.500, 10.500, 11.500]],
        tolerance=0.5,
        decimals=None,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 3, 6], [0, 3, 6]],
        expected_diffs=[[0.25, 0, 0.5], [0.25, 0, 0.5]],
        expected_precision=[0.375, 0.375],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.5454545454545454, 0.5454545454545454],
        expected_rval=[-0.422, -0.422],
    ),
    # ## default tolerance, precision=3 (happens to be default)
    # ### all hits
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001, 15.0001], [0.0001, 5.0001, 10.0001, 15.0001]],
        preds=[[0.0004, 5.0004, 10.0004, 15.0004], [0.0004, 5.0004, 10.0004, 15.0004]],
        tolerance=None,
        decimals=3,
        expected_hits_target=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_hits_preds=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_diffs=[[0., 0., 0., 0.], [0., 0., 0., 0.]],
        expected_precision=[1.0, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[1.0, 1.0],
        expected_rval=[0.999, 0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001, 15.0001], [0.0001, 5.0001, 10.0001, 15.0001]],
        preds=[[1.0001, 6.0001, 11.0001, 16.0001], [1.0001, 6.0001, 11.0001, 16.0001]],
        tolerance=None,
        decimals=3,
        expected_hits_target=[[], []],
        expected_hits_preds=[[], []],
        expected_diffs=[[], []],
        expected_precision=[0.0, 0.0],
        expected_recall=[0.0, 0.0],
        expected_fscore=[0.0, 0.0],
        expected_rval=[0.292, 0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[[1.0001, 6.0001, 10.0004, 16.0001], [1.0001, 6.0001, 10.0004, 16.0001]],
        preds=[[0.0001, 5.0001, 10.0001, 15.0001], [0.0001, 5.0001, 10.0001, 15.0001]],
        tolerance=None,
        decimals=3,
        expected_hits_target=[[2], [2]],
        expected_hits_preds=[[2], [2]],
        expected_diffs=[[0], [0]],
        expected_precision=[0.25, 0.25],
        expected_recall=[0.25, 0.25],
        expected_fscore=[0.25, 0.25],
        expected_rval=[0.359, 0.359],
    ),
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001, 15.0001], [0.0001, 5.0001, 10.0001, 15.0001]],
        preds=[[0.0004, 5.0004, 10.0004], [0.0004, 5.0004, 10.0004]],
        tolerance=None,
        decimals=3,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0, 0, 0], [0, 0, 0]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.75, 0.75],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.823, 0.823],
    ),
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001], [0.0001, 5.0001, 10.0001]],
        preds=[[0.0004, 5.0004, 10.0004, 15.0004], [0.0004, 5.0004, 10.0004, 15.0004]],
        tolerance=None,
        decimals=3,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0, 0, 0], [0, 0, 0]],
        expected_precision=[0.75, 0.75],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.715, 0.715],
    ),
    # ## tolerance of 0.5, decimals=3 (default)
    # ### all hits
    PRFRTestCase(
        target=[[0.0004, 5.0004, 10.0004, 15.0004], [0.0004, 5.0004, 10.0004, 15.0004]],
        preds=[[0.5001, 5.5001, 10.5001, 15.5001], [0.5001, 5.5001, 10.5001, 15.5001]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_hits_preds=[[0, 1, 2, 3], [0, 1, 2, 3]],
        expected_diffs=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        expected_precision=[1.0, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[1.0, 1.0],
        expected_rval=[0.999, 0.999],
    ),
    # ### no hits
    PRFRTestCase(
        target=[[0.0004, 5.0004, 10.0004, 15.0004], [0.0004, 5.0004, 10.0004, 15.0004]],
        preds=[[1.5001, 6.5001, 11.5001, 16.5001], [1.5001, 6.5001, 11.5001, 16.5001]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[], []],
        expected_hits_preds=[[], []],
        expected_diffs=[[], []],
        expected_precision=[0.0, 0.0],
        expected_recall=[0.0, 0.0],
        expected_fscore=[0.0, 0.0],
        expected_rval=[0.292, 0.292],
    ),
    # ### no > hits > all
    PRFRTestCase(
        target=[[0.0004, 5.0004, 10.0004, 15.0004], [0.0004, 5.0004, 10.0004, 15.0004]],
        preds=[[1.5001, 6.5001, 10.5001, 16.5001], [1.5001, 6.5001, 10.5001, 16.5001]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[2], [2]],
        expected_hits_preds=[[2], [2]],
        expected_diffs=[[0.5], [0.5]],
        expected_precision=[0.25, 0.25],
        expected_recall=[0.25, 0.25],
        expected_fscore=[0.25, 0.25],
        expected_rval=[0.359, 0.359],
    ),
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001, 15.0001], [0.0001, 5.0001, 10.0001, 15.0001]],
        preds=[[0.5004, 5.5004, 10.5004], [0.5004, 5.5004, 10.5004]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.75, 0.75],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.823, 0.823],
    ),
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001], [0.0001, 5.0001, 10.0001]],
        preds=[[0.5004, 5.5004, 10.5004, 15.5004], [0.5004, 5.5004, 10.5004, 15.5004]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[0.75, 0.75],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.715, 0.715],
    ),
    # ### multiple hits, tests we only keep one
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001], [0.0001, 5.0001, 10.0001]],
        preds=[[0.5004, 1.5004, 5.5004, 10.5004], [0.5004, 1.5004, 5.5004, 10.5004]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 2, 3], [0, 2, 3]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[0.75, 0.75],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 0.8571428571428571],
        expected_rval=[0.715, 0.715],
    ),
    PRFRTestCase(
        target=[[0.0001, 5.0001, 10.0001], [0.0001, 5.0001, 10.0001]],
        preds=[
            [0.2504, 0.5004, 2.5004, 5.0004, 5.5004, 7.5004, 10.5004, 11.5004], 
            [0.2504, 0.5004, 2.5004, 5.0004, 5.5004, 7.5004, 10.5004, 11.5004]
        ],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 3, 6], [0, 3, 6]],
        expected_diffs=[[0.25, 0, 0.5], [0.25, 0, 0.5]],
        expected_precision=[0.375, 0.375],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.5454545454545454, 0.5454545454545454],
        expected_rval=[-0.422, -0.422],
    ),
    # # edge cases
    # no boundaries in target
    PRFRTestCase(
        target=[[], []],
        preds=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[], []],
        expected_hits_preds=[[], []],
        expected_diffs=[[], []],
        expected_precision=[0.0, 0.0],
        expected_recall=[0.0, 0.0],
        expected_fscore=[0.0, 0.0],
        expected_rval=[0.0, 0.0],
    ),
    # no boundaries in preds
    PRFRTestCase(
        target=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        preds=[[], []],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[], []],
        expected_hits_preds=[[], []],
        expected_diffs=[[], []],
        expected_precision=[0.0, 0.0],
        expected_recall=[0.0, 0.0],
        expected_fscore=[0.0, 0.0],
        expected_rval=[0.0, 0.0],
    ),
    # only one boundary in ref/hyp
    PRFRTestCase(
        target=[[1.0], [1.0]],
        preds=[[1.0], [1.0]],
        tolerance=0.5,
        decimals=3,
        expected_hits_target=[[0], [0]],
        expected_hits_preds=[[0], [0]],
        expected_diffs=[[0], [0]],
        expected_precision=[1.0, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[1.0, 1.0],
        expected_rval=[0.999, 0.999],
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/119
    PRFRTestCase(
        target=[[2.244, 2.262], [2.244, 2.262]],
        preds=[[2.254], [2.254]],
        tolerance=0.01,
        decimals=3,
        expected_hits_target=[[1], [1]],
        expected_hits_preds=[[0], [0]],
        expected_diffs=[[0.008], [0.008]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.5, 0.5],
        expected_fscore=[
            (2 * 1.0 * 0.5) / (1 + 0.5),  # 0.6666666666666666 (repeating)
            (2 * 1.0 * 0.5) / (1 + 0.5),  # 0.6666666666666666 (repeating)
        ],
        expected_rval=[0.646, 0.646],
    ),
    # this is a regression test
    # see https://github.com/vocalpy/vocalpy/issues/170
    PRFRTestCase(
        target=[[], []],
        preds=[[], []],
        tolerance=None,
        decimals=None,
        expected_hits_target=[[], []],
        expected_hits_preds=[[], []],
        expected_diffs=[[], []],
        expected_precision=[1.0, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[1.0, 1.0],
        expected_rval=[1.0, 1.0],
    ),
    # test cases with `ignore_val`
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, -100.0]],
        preds=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        tolerance=None,
        decimals=None,
        ignore_val=-100,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0, 0, 0], [0, 0, 0]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.75, 1.0],
        expected_fscore=[0.8571428571428571, 1.0],
        expected_rval=[0.823, 1.0],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        preds=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, -100.0]],
        tolerance=None,
        decimals=None,
        ignore_val=-100.0,
        expected_hits_target=[[0, 1, 2],[0, 1, 2]],
        expected_hits_preds=[[0, 1, 2],[0, 1, 2]],
        expected_diffs=[[0, 0, 0],[0, 0, 0]],
        expected_precision=[0.75, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 1.0],
        expected_rval=[0.715, 1.0],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000, 15.000], [0.000, 5.000, 10.000, -100.0]],
        preds=[[0.500, 5.500, 10.500], [0.500, 5.500,10.500]],
        tolerance=0.5,
        decimals=None,
        ignore_val=-100.0,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[1.0, 1.0],
        expected_recall=[0.75, 1.0],
        expected_fscore=[0.8571428571428571, 1.0],
        expected_rval=[0.823, 1.0],
    ),
    PRFRTestCase(
        target=[[0.000, 5.000, 10.000], [0.000, 5.000, 10.000]],
        preds=[[0.500, 5.500, 10.500, 15.500], [0.500, 5.500, 10.500, -100.0]],
        tolerance=0.5,
        decimals=None,
        ignore_val=-100.0,
        expected_hits_target=[[0, 1, 2], [0, 1, 2]],
        expected_hits_preds=[[0, 1, 2], [0, 1, 2]],
        expected_diffs=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        expected_precision=[0.75, 1.0],
        expected_recall=[1.0, 1.0],
        expected_fscore=[0.8571428571428571, 1.0],
        expected_rval=[0.715, 1.0],
    ),
]
