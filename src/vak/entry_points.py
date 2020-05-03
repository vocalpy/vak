def _iter(group_name):
    """helper function that iterates through entry points, given name of entry point group"""
    try:
        import pkg_resources
    except (ImportError, IOError):
        return []

    return pkg_resources.iter_entry_points(group_name)
