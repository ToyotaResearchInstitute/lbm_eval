from collections import OrderedDict
import copy
from functools import partial
import inspect
from textwrap import dedent, indent

import numpy as np

from pydrake.common.value import AbstractValue, Value
from pydrake.systems.framework import (
    BasicVector,
    BasicVector_,
    Context,
    LeafSystem,
    PortDataType,
    SystemScalarConverter,
)

from anzu.sim.common.primitive_systems import is_ndarray_like

SCALAR_TYPES = SystemScalarConverter.SupportedScalars

# N.B. For vectors, I (Eric) explicitly want to avoid conflicting with these
# possible future types:
#  https://github.com/RobotLocomotion/drake/pull/13160
#  https://github.com/RobotLocomotion/drake/blob/6f1927f1d8b7cb72eb553ee9bc151840dc8ebae4/bindings/pydrake/common/_value_extra.py


def _is_typish(obj):
    # This is a bit bad :(
    # TODO(eric.cousineau): Consolidate Anzu's `typing_` to use `typeish`, and
    # then propagate that same info to Drake so that this hack can be removed.
    if repr(obj).startswith("List["):
        return True
    elif isinstance(obj, type):
        return True
    else:
        return False


class VectorArg:
    """Defines annotation for a vector of a given size."""
    # TODO(eric.cousineau): Use NDArray from typing_ at some point.
    def __init__(self, size):
        self._size = size

    def __repr__(self):
        return f"VectorArg({self._size})"


class _ContextTimeArg:
    """Defines an annotation that implies time from the context should be used
    for a function argument."""
    def __repr__(self):
        return "ContextTimeArg"


ContextTimeArg = _ContextTimeArg()


class _MultiOutputType:
    """Annotation class object for a multi-output function that dictates
    argument names and types."""
    def __init__(self, items):
        if isinstance(items, OrderedDict):
            self._annotations = items
        elif isinstance(items, (tuple, list)):
            self._annotations = OrderedDict(items)
        else:
            raise RuntimeError("Must use tuple, list, or OrderedDict")
        assert len(self._annotations) > 0, (
            "MultiOutput must be of non-zero length.")

    def __repr__(self):

        def as_str(annotation):
            if isinstance(annotation, type):
                return annotation.__name__
            else:
                return str(annotation)

        items_str = [
            f"{repr(key)}: {as_str(value)}"
            for key, value in self._annotations.items()]
        return f"MultiOutput[{', '.join(items_str)}]"


class _MultiOutputSugar:
    """Allows multiple outputs to be defined using the following syntax::

        MultiOutput['arg1': Type1, 'arg2': Type2, ...]
    """

    def _slice_to_item(self, s):
        # self['arg1': Type1] will contain the following:
        #   slice(start='arg1', stop=Type1, step=None)
        assert isinstance(s, slice)
        assert s.step is None, (
            "You cannot use the syntax ['arg1':Something:Type]")
        name = s.start
        annotation = s.stop
        return (name, annotation)

    def __getitem__(self, slices):
        if slices is None:
            slices = ()
        elif not isinstance(slices, tuple):
            slices = (slices,)
        items = [self._slice_to_item(s) for s in slices]
        return _MultiOutputType(items)


# Define user-friendly object. See docstring for `_MultiOutputSugar`.
MultiOutput = _MultiOutputSugar()


# TODO(eric.cousineau): Support declaring / using state, and handle naming.


def _get_value_inner_cls(value_cls):
    assert value_cls is not AbstractValue, "Need concrete instantiation"
    assert issubclass(value_cls, AbstractValue), value_cls
    # Assume there is only one instantiation registered to this class.
    (inner_cls,), = Value.get_param_set(value_cls)
    return inner_cls


def _is_basic_vector(cls):
    return BasicVector_.is_instantiation(cls)


class Factory:
    def __init__(self, cls, *args, **kwargs):
        if callable(cls):
            self._func = partial(cls, *args, **kwargs)
        else:
            self._func = lambda: copy.copy(cls)

    def __call__(self):
        return self._func()


def _get_abstract_model_and_example(cls):
    # Permit for annotations, which should generally be types. Returns
    # (model, example), where `model` should be of type `AbstractValue`.
    example = None

    if isinstance(cls, Factory):
        example = cls()
        cls = type(example)

    assert _is_typish(cls), f"Must supply a type: {cls}"
    if isinstance(cls, type) and issubclass(cls, AbstractValue):
        # User wants to explicitly deal with abstract values.
        inner_cls = _get_value_inner_cls(cls)
        assert not _is_basic_vector(inner_cls), (
            f"Cannot specify Value[BasicVector_[]]; specify "
            "BasicVector_[](size) instead: {cls}")
        if example is None:
            example = cls()
        model = example
        return model, example
    else:
        if example is None:
            example = cls()
        model = AbstractValue.Make(example)
        return model, example


def _is_vector(x):
    # Determines if x is a 1D or 2D vector.
    return (x.ndim in (1, 2) and x.size in x.shape)


def value_to_annotation(value):
    """
    Converts a value to a corresponding annotation:
    - If it's an array, return VectorArg(size).
    - If it's a BasicVector instance, return it itself.
    - If it's an AbstractValue and of type Value[BasicVector_[]], return the
    inner value.
    - Otherwise, return the class directly.
    """
    assert not _is_typish(value), (
        f"Must supply value, not type: {value}")
    cls = type(value)
    if value is None:
        return None
    elif is_ndarray_like(value):
        value = np.asarray(value)
        if value.ndim == 0:
            # Scalar.
            return type(value.item())
        else:
            assert _is_vector(value), (
                f"Numpy arrays must be vectors: {value}")
            return VectorArg(value.size)
    elif BasicVector_.is_instantiation(cls):
        return value
    elif isinstance(value, AbstractValue):
        # TODO(eric.cousineau): How to handle `VectorBase` inheritance? (blech)
        if _is_basic_vector(_get_value_inner_cls(cls)):
            return value.get_value()
        else:
            return cls
    else:
        return cls


class _ArgHelper:
    """Provides information and functions to aid in interfacing a Python
    function with the Systems framework."""
    def __init__(self, name, cls, scalar_as_vector):
        """Given a class (or type annotation), figure out the type (vector
        port, abstract port, or context time), the model value (for ports), and
        example value (for output inference)."""

        # Name can be overridden.
        self.name = name
        self._scalar_needs_conversion = False
        self._is_direct_type = False
        if isinstance(cls, VectorArg):
            self.type = PortDataType.kVectorValued
            self.model = BasicVector(cls._size)
            self.model.get_mutable_value()[:] = 0
            self.example = self.model.get_value()
        elif BasicVector_.is_instantiation(cls):
            assert False, (
                f"Must supply BasicVector_[] instance, not type: {cls}")
        elif BasicVector_.is_instantiation(type(cls)):
            self.type = PortDataType.kVectorValued
            self.model = cls
            self.example = self.model
            self._is_direct_type = True
        elif scalar_as_vector and cls in SCALAR_TYPES:
            self.type = PortDataType.kVectorValued
            self.model = BasicVector(1)
            self.model.get_mutable_value()[:] = 0
            self.example = float()  # Should this be smarter about the type?
            self._scalar_needs_conversion = True
        elif cls is ContextTimeArg:
            self.type = ContextTimeArg
            self.model = None
            self.example = float()
        else:
            self.type = PortDataType.kAbstractValued
            self.model, self.example = _get_abstract_model_and_example(cls)
            if self.model is self.example:
                self._is_direct_type = True

    def _squeeze(self, x):
        if self._scalar_needs_conversion:
            assert x.shape == (1,), f"Bad input: {x}"
            return x.item(0)
        else:
            return x

    def _unsqueeze(self, x):
        if self._scalar_needs_conversion:
            return np.array([x])
        else:
            return x

    def declare_input_eval(self, system):
        """Declares an input evaluation function. If a port is needed, will
        declare the port."""
        if self.type is ContextTimeArg:
            return Context.get_time
        elif self.type == PortDataType.kAbstractValued:
            # DeclareInputPort does not work with kAbstractValued :(
            port = system.DeclareAbstractInputPort(
                name=self.name, model_value=self.model)
            if self._is_direct_type:
                return port.EvalAbstract
            else:
                return port.Eval
        else:
            port = system.DeclareVectorInputPort(
                name=self.name, model_vector=self.model)
            if self._is_direct_type:
                return port.EvalBasicVector
            else:
                return lambda context: self._squeeze(port.Eval(context))

    def declare_output_port(self, system, calc):
        """Declares an output port on a given system."""
        if self.type is ContextTimeArg:
            assert False, dedent(r"""\
                ContextTimeArg is disallowed for output arguments. If needed,
                explicitly pass it through, e.g.:
                    def context_time(t: ContextTimeArg):
                        return t
                """)
        elif self.type == PortDataType.kAbstractValued:
            system.DeclareAbstractOutputPort(
                name=self.name, alloc=self.model.Clone, calc=calc)
        else:
            system.DeclareVectorOutputPort(
                name=self.name, model_value=self.model, calc=calc)

    def get_set_output_func(self):
        assert self.type is not ContextTimeArg
        if self.type == PortDataType.kAbstractValued:
            if self._is_direct_type:
                return lambda output, value: output.SetFrom(value)
            else:
                return lambda output, value: output.set_value(value)
        else:
            if self._is_direct_type:
                # TODO(eric.cousineau): Bind VectorBase.SetFrom().
                return lambda output, value: output.SetFromVector(
                    value.get_value())
            else:
                return lambda output, value: output.SetFromVector(
                    self._unsqueeze(value))


class _FunctionDeclaration:
    """
    Declares the necessary inputs and outputs for a system given a function.
    """
    def __init__(self, system, func, scalar_as_vector=True):
        self._system = system
        self._func = func
        self._scalar_as_vector = scalar_as_vector
        self._publish_func = None
        signature = inspect.signature(self._func)
        input_helpers = self._get_input_helpers(signature)
        self._declare_inputs(input_helpers)
        self._declare_computation(signature)
        self._fail_if_recurse = False

    def _make_arg_helper(self, name, value):
        return _ArgHelper(name, value, self._scalar_as_vector)

    def _call(self, context):
        # Calls user function given the context.
        return self._func(**self._kwargs(context))

    def _get_input_helpers(self, signature):
        # Takes a function signature and produces a set of input helpers.
        accepted_kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        errors = []
        input_helpers = []
        for name, param in signature.parameters.items():
            if param.kind not in accepted_kinds:
                errors.append(f"Invalid kind: {param}")
                continue
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                errors.append(f"Needs annotation: {param}")
                continue
            input_helpers.append(self._make_arg_helper(name, annotation))
        if errors:
            raise RuntimeError(
                f"Errors for '{self._func.__name__}':\n" +
                indent("\n".join(errors), "  "))
        return input_helpers

    def _declare_inputs(self, input_helpers):
        """Declares inputs necessary to call the function."""
        self._kwargs_eval = dict()
        self._kwargs_example = dict()
        for input_helper in input_helpers:
            input_eval = input_helper.declare_input_eval(self._system)
            self._kwargs_eval[input_helper.name] = input_eval
            self._kwargs_example[input_helper.name] = input_helper.example

    def _declare_computation(self, signature):
        """Declares outputs or publish function."""
        output_cls = signature.return_annotation
        if output_cls is inspect.Parameter.empty:
            # Infer via evaluation, only if needed.
            output_value = self._func(**self._kwargs_example)
            # Disable this because it's confusing to see side effects.
            assert output_value is not None, (
                f"For function '{self._func.__name__}': 'None' is not a valid "
                f"inferred output. "
                f"Please annotate the output as either '-> None' (if you are "
                f"just printing) or '-> object' (if None is valid, but not "
                f"the only choice).")
            output_cls = value_to_annotation(output_value)
        if isinstance(output_cls, _MultiOutputType):
            self._declare_multi_output(output_cls._annotations)
        elif output_cls is not None:
            self._declare_single_output(output_cls)
        else:
            self._declare_publish_only()

    def _declare_publish_only(self):
        # Use (declare?) publish.
        self._publish_func = self._call

    def _declare_single_output(self, output_cls):
        # Single output port.
        output_helper = self._make_arg_helper("output", output_cls)
        set_output = output_helper.get_set_output_func()

        def calc_output(context, output):
            set_output(output, self._call(context))

        output_helper.declare_output_port(
            self._system, calc=calc_output)

    def _declare_multi_output(self, annotations):
        # Multiple output ports.
        num_outputs = len(annotations)

        def calc_output_i(context, output_i, i, set_output, name):
            # TODO(eric.cousineau): Use caching to consolidate output?
            try:
                value = self._call(context)
            except RecurseError as e:
                raise RecurseError(f"For output {name}: {e}")

            assert isinstance(value, tuple), (
                f"Output '{name}' type must be a tuple. Got "
                f"{type(value)} instead.")
            assert len(value) == num_outputs, (
                f"Output '{name}' must be tuple of length {num_outputs}. "
                f"Got length {len(value)} instead.")
            set_output(output_i, value[i])

        for i, (name, output_cls) in enumerate(annotations.items()):
            output_helper = self._make_arg_helper(name, output_cls)
            set_output = output_helper.get_set_output_func()
            calc = partial(
                calc_output_i, i=i, set_output=set_output, name=name)
            output_helper.declare_output_port(self._system, calc=calc)

    def _kwargs(self, context):
        # TODO(eric.cousineau): Declare as direct feedthrough instead...
        if self._fail_if_recurse:
            raise InnerError()
        self._fail_if_recurse = True
        try:
            kwargs = dict()
            for name, input_eval in self._kwargs_eval.items():
                try:
                    kwargs[name] = input_eval(context)
                except InnerError:
                    raise RecurseError(f"Recursing on eval of '{name}'")

            return kwargs
        finally:
            self._fail_if_recurse = False

    def maybe_publish(self, context):
        # TODO(eric.cousineau): Better way to aggregate publish functions via
        # the systems framework? (e.g. declare the event?)
        if self._publish_func is not None:
            self._publish_func(context)


class InnerError(Exception):
    pass


class RecurseError(RuntimeError):
    pass


# TODO(eric.cousineau): Support scalar conversion.


class FunctionSystem(LeafSystem):
    """Creates a function with inputs and outputs specified by type
    annotations.

    Examples::

        def print_only() -> None:
            print("Hello")

        def add(x: VectorArg(3), y: VectorArg(3)):
            return x + y

        def multi_output() -> MultiOutput['x': VectorArg(3), 's': str]:
            x = [1., 2., 3.]
            s = "hello world"
            return x, s

        # This will have zero inputs and outputs. Since there are no
        # outputs, this print function will be triggered by a publish
        # event.
        FunctionSystem(print_only)

        # This will have two inputs and one output. The function will be
        # called when the output port is queried.
        # Because this has no output annotation, ->, this function
        # will be evaluated to infer the appropriate type.
        FunctionSystem(add)

        # This will have zero inputs, but 2 outputs.
        FunctionSystem(multi_output)
    """
    def __init__(self, funcs, scalar_as_vector=True, set_name=True, *,
                 publish_period=None):
        """
        Arguments:
            func: Function with necessary annotations. If the output is not
                annotated, then the function will be called with sample
                arguments. This may cause unwanted side-effects, especially
                when printing, so you should consider annotating the output
                type.
            scalar_as_vector: If True, inputs and outputs that are of
                (float, AutoDiffXd, Expression) will interpreted as
                VectorArg(1).
        """
        LeafSystem.__init__(self)
        if not isinstance(funcs, (list, tuple)):
            funcs = (funcs,)
        assert len(funcs) > 0
        cls = type(self)
        if set_name:
            func_names = ", ".join([
                f"{func.__name__}"
                for func in funcs])
            self.set_name(f"{cls.__name__}[{func_names}]")
        # TODO(eric.cousineau): Support sharing inputs with multiple functions.
        self._func_declaration = [
            _FunctionDeclaration(self, func, scalar_as_vector=scalar_as_vector)
            for func in funcs]
        self.DeclareForcedPublishEvent(publish=self._on_publish)
        if publish_period is not None:
            self.DeclarePeriodicPublishEvent(
                period_sec=publish_period, offset_sec=0.0,
                publish=self._on_publish)

    def _on_publish(self, context):
        for func_declaration in self._func_declaration:
            func_declaration.maybe_publish(context)


def make_printer(cls, print=print, *, publish_period=None):
    """Creates a system which prints the value of its input port "value".
    for a given `cls`. Usages of this should still ensure a publish period is
    declared. Example:

        printer = builder.AddSystem(make_printer(str))
        printer.DeclarePeriodicPublish(dt)
    """

    def print_value(value: cls) -> None:
        print(value)

    printer = FunctionSystem(print_value, set_name=False,
                             publish_period=publish_period)
    return printer


def connect_printer(builder, output, print=print, use_direct_type=False, *,
                    publish_period=None):
    model = output.Allocate()
    if use_direct_type:
        cls = value_to_annotation(model)
    else:
        value = model.get_value()
        if _is_basic_vector(type(value)):
            value = value.get_value()
        cls = value_to_annotation(value)
    printer = builder.AddSystem(make_printer(cls, print=print,
                                             publish_period=publish_period))
    builder.Connect(output, printer.get_input_port(0))
    return printer
