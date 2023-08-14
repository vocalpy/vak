def get_train_dur_replicate_split_name(
    train_dur: int, replicate_num: int
) -> str:
    """Get name of a training set split for a learning curve,
    for a specified training duration and replicate number.

    Used when preparing the training set splits for a learning curve,
    and when training models to generate the results for the curve.
    """
    return f"train-dur-{float(train_dur)}-replicate-{int(replicate_num)}"
