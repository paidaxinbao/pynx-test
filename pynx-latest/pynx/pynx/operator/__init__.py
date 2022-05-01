# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2017-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import numpy as np


class OperatorException(Exception):
    pass


class Operator(object):
    """
    Base class for an operator, applying e.g. to a wavefront object.

    TODO: define this a a proper python abstract base class.
    """

    def __init__(self, lazy=False):
        if self.__class__ is not Operator:
            self.ops = [self]
        else:
            self.ops = []

        # We keep this to be derived for specialization (e.g. CLOperatorWavefront,..)
        self.Operator = Operator
        self.OperatorSum = OperatorSum
        self.OperatorPower = OperatorPower

        # If this flag is set, the operator will not be directly applied to the object,
        # but will wait for the next (non-lazy) operation. This can be useful
        # when preparing an object and storing operations which can only
        # be applied once the object is fully ready
        self.lazy = lazy

    def __mul__(self, w):
        """
        Applies the operator to an object. 

        :param w: the object to which this operator will be applied. 
                    If it is another Operator, the operation will be stored for later application.
        :return: the object, result of the operation. Usually the same object (modified) as w.
        """
        if np.isscalar(w):
            if np.isclose(w, 1):
                return self
            o = self.Operator()
            o.ops = self.ops + [w]
            return o
        if isinstance(w, OperatorSum) or isinstance(w, OperatorPower):
            o = self.Operator()
            o.ops = self.ops + [w]
            return o
        elif isinstance(w, Operator):
            o = self.Operator()
            o.ops = self.ops + w.ops
            return o
        self.apply_ops_mul(w)
        return w

    def __rmul__(self, x):
        """
        Applies the operator to an object.

        :param w: the left-hand object. This can only be a scalar, e.g. for 5*Op()
        :return: the object, result of the operation. Usually the same object (modified) as w.
        """
        if np.isscalar(x) is False:
            raise OperatorException("ERROR: attempted Op1 * Op2, with Op1=%s, Op2=%s" % (str(x), str(self)))
        o = self.Operator()
        o.ops = [x] + self.ops
        return o

    def __pow__(self, power):
        """

        :param power: a strictly positive integer
        :param modulo: 
        :return: 
        """
        assert isinstance(power, int) or isinstance(power, np.integer)
        return self.OperatorPower(self, power)

    def __add__(self, rhs):
        """
        Add two operators, creating a new Operator object. When applied to an object, this will  a temporary
        copy of the object data to apply operators in the sum.

        This is not yet functional !

        :param rhs: another Operator, or a scalar.
        :return: A new Operator object which represents the sum of Operators
        """
        return self.OperatorSum(self, rhs)

    def __radd__(self, lhs):
        """
        Add two operators, creating a new Operator object. When applied to an object, this will use a temporary
        copy of the object data to apply operators in the sum.

        :param lhs: another Operator, or a scalar.
        :return: A new Operator object which represents the sum of Operators
        """
        return self + lhs

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        """
        Multiply an operator by -1
        :return:
        """
        return -1 * self

    def __str__(self):
        """
        :return: a string description of the operator
        """
        s = ""
        for o in reversed(self.ops):
            if o == self:
                n = self.__class__.__name__ + "()"
                if has_attr_not_none(self, 'nb_cycle'):
                    if isinstance(self.nb_cycle, (int, np.integer)):
                        if self.nb_cycle > 1:
                            n += "**%d" % (self.nb_cycle)
            else:
                n = str(o)
            if isinstance(o, OperatorSum):
                n = "(" + n + ")"
            if s == "":
                s = n
            else:
                s = n + "*" + s
        return s

    def apply_ops_mul(self, w):
        """
        Apply the series of operators stored in self.ops to an object. 
        The operators are applied one after the other to the same object (multiplication), which
        is modified in-place each time.

        :param w: the object to which the operators will be applied.
        :return: the object, after application of all the operators in sequence
        """
        if self.lazy:
            # The operation will be applied with a delay to the object,
            # during the next non-lazy operation
            self.lazy = False
            # print("Storing operation for lazy evaluation:", str(self))
            if has_attr_not_none(w, "_lazy_ops"):
                w._lazy_ops.insert(0, self)
            else:
                w._lazy_ops = [self]
            return w
        # First check there are no lazy operations waiting
        if has_attr_not_none(w, "_lazy_ops"):
            if len(w._lazy_ops):
                # s = "Applying lazy operations:"
                # for o in reversed(w._lazy_ops):
                #     s += str(o)
                # print(s)
                self.ops = self.ops + w._lazy_ops
                w._lazy_ops = []
        # Apply the chain of operation
        for o in reversed(self.ops):
            if np.isscalar(o):
                w = o * w
            else:
                o.prepare_data(w)
                w = o.op(w)
                o.timestamp_increment(w)
        return w

    def op(self, w):
        """
        Applies the operator to one object.
        Virtual function, must be derived. By default this is the identity operator.

        :return: the result of the operation, usually the same type as the input, which is modified in-place.
            But this can also be a scalar (reduction).
        """
        return w

    def prepare_data(self, w):
        """
        Make sure the data to be used is in the correct memory (host or GPU) for the operator. 
        Virtual, must be derived.
        
        :param w: the object (e.g. wavefront) the operator will be applied to.
        :return: 
        """
        pass

    def timestamp_increment(self, w):
        """
        Increment the timestamp counter corresponding to the processing language used (OpenCL or CUDA)
        Virtual, must be derived.

        :param w: the object (e.g. wavefront) the operator will be applied to.
        :return: 
        """
        pass

    def view_register(self, obj):
        """
        Creates a new unique view key in an object. When finished with this view, it should be de-registered
        using view_purge. This also create an empty view (None).
        :return: an integer value, which corresponds to yet-unused key in the object's view.
        """
        raise OperatorException("ERROR: attempted to apply Operator.view_register() which is pure virtual")

    def view_copy(self, obj, i_source, i_dest):
        """
        Create a new view of the object by copying the original data. This will make a copy of all relevant data,
        which can be a wavefront, CDI object, Ptychography object, probe and psi arrais, etc...

        This (virtual) function is used to make temporary copies of the data Operators apply to. This is used
        to make linear combination (additions) of operators, which requires several copies of the data.
        As the copying part depends on the processing unit used (GPU, CPU) and the exact data to duplicate, this
        is a pure virtual class, which must be derived to be used.
        Note:
            - this should only copy the 'active' data (which is affected by calculations)
            - index 0 corresponds to the original array, to which subsequent operators will be applied to

        :param obj: the object where the data will be duplicated
        :param i_source: the index (integer) of the source object data
        :param i_dest: the index (integer) of the destination object data
        :return: nothing. The object is modified in-place
        """
        raise OperatorException("ERROR: attempted to apply Operator.view_copy() which is pure virtual")

    def view_swap(self, obj, i1, i2):
        """
        Swap the object view between index i1 and i2.
        As the swapping part depends on the processing unit used (GPU, CPU) and the exact data to swap, this
        is a pure virtual function, which must be derived to be used.
        :param obj: the object where the data will be duplicated
        :param i1: the index (integer) of the first object data
        :param i2: the index (integer) of the second object data
        :return: nothing. The object is modified in-place
        """
        raise OperatorException("ERROR: attempted to apply Operator.view_swap() which is pure virtual")

    def view_sum(self, obj, i_source, i_dest):
        """
        Add the view data from one index into another.
        As the summing depends on the processing unit used (GPU, CPU) and the exact data to sum, this
        is a pure virtual function, which must be derived to be used.
        :param obj: the object where the data will be duplicated
        :param i_source: the index (integer) of the source object data
        :param i_dest: the index (integer) of the destination object data
        :return: nothing. The object is modified in-place
        """
        raise OperatorException("ERROR: attempted to apply Operator.view_sum() which is pure virtual")

    def view_purge(self, obj, i_view):
        """
        Purge the different views of an object (except the main one).
        As the purging depends on the processing unit used (GPU, CPU) and the exact data to purge, this
        is a pure virtual function, which must be derived to be used.
        :param obj: the object where the view will be purged
        :param i_view: the index of the view to purge. If None, all views are purged. This de-registers the view.
        :return: nothing
        """
        raise OperatorException("ERROR: attempted to apply Operator.view_purge() which is pure virtual")


class OperatorPower(Operator):
    """
    Operator class for Operator**N, applying N time the operation.
    """

    def __init__(self, op, n):
        super(OperatorPower, self).__init__()
        self.oper = op
        self.n = n

    def __pow__(self, power):
        """
        :param power: a strictly positive integer
        :return:
        """
        assert isinstance(power, int) or isinstance(power, np.integer)
        o = OperatorPower(self.oper, self.n ** power)
        return o

    def __str__(self):
        """
        :return: a string description of the operator
        """
        if isinstance(self.oper, OperatorSum):
            return "(%s)**%d" % (str(self.oper), self.n)
        return "%s**%d" % (str(self.oper), self.n)

    def op(self, w):
        """
        Apply the series of operators stored in self.oper to an object, repeated self.n times.

        :param w: the object to which the operator will be applied.
        :return: the object, after applying the operator self.n times.
        """

        for i in range(self.n):
            w = self.oper * w

        return w


class OperatorSum(Operator):
    """
    Operator class for a sum of Operators.
    """

    def __init__(self, op1, op2):
        super(OperatorSum, self).__init__()
        self.ops = [op1, op2]

    def __add__(self, other):
        if isinstance(other, Operator) or np.isscalar(other):
            self.ops += [other]
            return self
        else:
            raise OperatorException("ERROR: Cannot add %s and %s" % (str(self), str(other)))

    def __radd__(self, other):
        if isinstance(other, Operator) or np.isscalar(other):
            self.ops = [other] + self.ops
            return self
        else:
            raise OperatorException("ERROR: Cannot add %s and %s" % (str(self), str(other)))

    def __str__(self):
        """
        :return: a string description of the operator
        """
        s = ""
        for o in reversed(self.ops):
            if o == self:
                n = self.__class__.__name__ + "()"
            else:
                n = str(o)
            if s == "":
                s = n
            else:
                s = n + "+" + s
        return s

    def op(self, o):
        """
        Apply the series of operators stored in self.ops to an object.
        The operators must be applied separately (using temporary copies) to the original object, and then summed.
        Virtual function, must be derived, as the copying method is GPU-dependant.

        This is not yet functional !

        :param o: the object to which the operators will be applied.
        :return: the object, as the sum of all the object produced by each operator in the sum.
        """
        if len(self.ops) == 2:
            i1 = self.view_register(o)
            # We need 2 copies of the data
            self.view_copy(o, 0, i1)
            o = self.ops[0] * o
            self.view_swap(o, 0, i1)
            o = self.ops[1] * o
            self.view_sum(o, i1, 0)
        else:
            i1 = self.view_register(o)
            i2 = self.view_register(o)
            # We need 3 copies of the data
            self.view_copy(o, 0, i1)
            o = self.ops[0] * o
            self.view_swap(o, 0, i2)
            for op in self.ops[1:]:
                self.view_copy(o, i1, 0)
                o = op * o
                self.view_sum(o, 0, i2)
            self.view_swap(o, 0, i2)
            self.view_purge(o, i2)
        self.view_purge(o, i1)
        return o


def has_attr_not_none(o, name):
    """
    Test the existence of an attribute in an object.

    :param o: the object
    :param name: the name of the attribute to be found
    :return: True if the attribute has been found, False otherwise
    """
    if name in dir(o):
        if o.__getattribute__(name) is not None:
            return True
    return False
