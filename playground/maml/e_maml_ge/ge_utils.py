import mock
import tensorflow as tf
from typing import Callable, Any, List, TypeVar


def as_dict(c):
    return {k: v for k, v in vars(c).items() if k[0] != "_"}


def var_like(var):
    name, dtype, shape = var.name, var.dtype, tuple(var.get_shape().as_list())
    new_name = name.split(':')[0]
    # note: assuming that you are using a variable scope for this declaration.
    new_var = tf.Variable(initial_value=tf.zeros(shape, dtype), name=new_name)
    # print(f"declaring variable like {name} w/ new name: {new_var.name}")
    return new_var


def placeholders_from_variables(var, name=None):
    """Returns a nested collection of TensorFlow placeholders that match shapes
    and dtypes of the given nested collection of variables.
    Arguments:
    ----------
        var: Nested collection of variables.
        name: Placeholder name.
    Returns:
    --------
        Nested collection (same structure as `var`) of TensorFlow placeholders.
    """
    if isinstance(var, list) or isinstance(var, tuple):
        result = [placeholders_from_variables(v, name) for v in var]
        if isinstance(var, tuple):
            return tuple(result)
        return result
    else:
        dtype, shape = var.dtype, tuple(var.get_shape().as_list())
        return tf.placeholder(dtype=dtype, shape=shape, name=name)


def wrap_variable_creation(func, custom_getter):
    """Provides a custom getter for all variable creations."""
    original_get_variable = tf.get_variable

    def custom_get_variable(*args, **kwargs):
        if hasattr(kwargs, "custom_getter"):
            raise AttributeError("Custom getters are not supported for optimizee variables.")
        return original_get_variable(
            *args, custom_getter=custom_getter, **kwargs)

    # Mock the get_variable method.
    with mock.patch("tensorflow.get_variable", custom_get_variable):
        return func()


T = TypeVar('T')


def make_with_custom_variables(func: Callable[[Any], T], variables: List[Any]) -> T:
    """Calls func and replaces any trainable variables.
    This returns the output of func, but whenever `get_variable` is called it
    will replace any trainable variables with the tensors in `variables`, in the
    same order. Non-trainable variables will re-use any variables already
    created.
    Arguments:
    ----------
        func: Function to be called.
        variables: A list of tensors replacing the trainable variables.
    Returns:
    --------
        The return value of func is returned.
    """
    index, n = -1, len(variables)

    def custom_getter(getter, name, **kwargs):
        nonlocal index
        index += 1
        return variables[index % n]

    return wrap_variable_creation(func, custom_getter)


# noinspection PyPep8Naming
class defaultlist():
    """allow using -1, -2 index to query from the end of the list, which is not possible with `defaultdict`. """

    def __init__(self, default_factory):
        self.data = list()
        self.default_factory = default_factory if callable(default_factory) else lambda: default_factory

    def __setitem__(self, key, value):
        try:
            self.data[key] = value
        except IndexError:
            self.data.extend([self.default_factory()] * (key + 1 - len(self.data)))
            self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __setstate__(self, state):
        raise NotImplementedError('need to be implemented for remote execution.')

    def __getstate__(self):
        raise NotImplementedError('need to be implemented for remote execution.')


def cache_ops(variables):
    cache = [var_like(v) for v in variables]
    # from pprint import pprint
    # old_vars = set(tf.global_variables())
    # new_vars = set(tf.global_variables()) - old_vars
    # pprint(new_vars)
    # pprint(cache)
    save = tf.group(*[tf.stop_gradient(c.assign(v)) for c, v in zip(cache, variables)])
    load = tf.group(*[tf.stop_gradient(v.assign(c)) for c, v in zip(cache, variables)])
    return save, load, cache
