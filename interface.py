# Credit: Pedro Matiello, https://github.com/pmatiello/python-interface

from inspect import getargspec #@UnresolvedImport

class interface(object):
    pass


class method(object):
    def __init__(self, args=None, varargs=None, keywords=None, defaults=0):
        self.args = args or []
        self.varargs = varargs
        self.keywords = keywords
        self.defaults = defaults

    def __str__(self):
        return "(%s,%s,%s)" % (self.args, self.varargs, self.keywords)


class implements(object):
    
    def __init__(self, interface):
        self.interface = interface
    
    def __call__(self, clazz):
        methods = [each for each in dir(self.interface) if self._is_method(each)]
        for each in methods:
            self._assert_implements(clazz, each)
        return clazz
    
    def _is_method(self, name):
        try:
            return type(self._attribute(self.interface, name)) == method
        except:
            False

    def _assert_implements(self, clazz, method_name):
        contract = self._attribute(self.interface, method_name)
        self._assert_method_presence(clazz, method_name, contract)
        method_impl = getargspec(self._attribute(clazz, method_name))
        self._assert_method_arguments(clazz, method_name, method_impl, contract)
        self._assert_method_varargs(clazz, method_name, method_impl, contract)
        self._assert_method_keyword_args(clazz, method_name, method_impl, contract)
        self._assert_method_default_args(clazz, method_name, method_impl, contract)

    def _assert_method_presence(self, clazz, method_name, contract):
        if (method_name not in dir(clazz)):
            raise InterfaceNotImplemented(self.interface, clazz, method_name, contract)
    
    def _assert_method_arguments(self, clazz, method_name, method_impl, contract):
        if (not contract.args == method_impl.args):
            raise InterfaceNotImplemented(self.interface, clazz, method_name, contract)
        
    def _assert_method_varargs(self, clazz, method_name, method_impl, contract):
        if (not contract.varargs == method_impl.varargs):
            raise InterfaceNotImplemented(self.interface, clazz, method_name, contract)

    def _assert_method_keyword_args(self, clazz, method_name, method_impl, contract):
        if (not contract.keywords == method_impl.keywords):
            raise InterfaceNotImplemented(self.interface, clazz, method_name, contract)

    def _assert_method_default_args(self, clazz, method_name, method_impl, contract):
        if (method_impl.defaults is None):
            defaults = 0
        else:
            defaults = len(method_impl.defaults)
        if (not contract.defaults == defaults):
            raise InterfaceNotImplemented(self.interface, clazz, method_name, contract)

    def _attribute(self, clazz, attribute):
        return object.__getattribute__(clazz, attribute)


class InterfaceNotImplemented(Exception):
    
    def __init__(self, interface, clazz, method_name, method_signature):
        self.interface = interface
        self.clazz = clazz
        self.method_name = method_name
        self.method_signature = method_signature
    
    def __str__(self):
        return "Class %s must implement method %s with arguments %s as defined in interface %s" % (self.clazz.__name__, self.method_name, self.method_signature, self.interface.__name__)
