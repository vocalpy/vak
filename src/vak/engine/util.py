"""utility functions specific to engine sub-package"""
BUILTIN_TYPES = [dict, list, str, set, tuple]

# define this outside Model class, so sub-classes can use it too
def _check_required_attribs(self, cls):
    """helper function that checks whether subclass declares attributes required by supraclass

    Parameters
    ----------
    self : Model
        or Model subclass that calls this function
        to check that a subclass of it has required attributes
    cls : type
        subclass of self, that must have the required class attributes
    """
    if not hasattr(self, 'REQUIRED_SUBCLASS_ATTRIBUTES'):
        raise AttributeError(
            f'Unable to validate whether {cls.__name__} has required attributes, '
            f'because {self.__name__} does not have the class variable "REQUIRED_SUBCLASS_ATTRIBUTES"'
        )
    for required_subclass_attr, subclass_attr_type in self.REQUIRED_SUBCLASS_ATTRIBUTES.items():
        if not hasattr(cls, required_subclass_attr):
            raise AttributeError(
                f'A subclass of {self.__name__} must declare a {required_subclass_attr},'
                f' but {cls.__name__} does not have an attribute {required_subclass_attr}.'
            )

        attr = getattr(cls, required_subclass_attr)
        if subclass_attr_type in BUILTIN_TYPES:
            # if it's required attr has a built-in data type, not a custom Python code class,
            # check if the attr is an instance of that built-in data type
            if type(attr) != subclass_attr_type:
                raise TypeError(
                    f'type of {attr} should be {subclass_attr_type} but was {type(attr)}'
                )
        else:
            if not issubclass(attr, subclass_attr_type):
                raise TypeError(
                    f'{required_subclass_attr} must be a subclass of {subclass_attr_type}, but '
                    f'type was: {type(attr)}'
                )
