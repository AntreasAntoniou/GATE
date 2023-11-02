from functools import wraps

from gate.models.core import is_desired_method, is_desired_variable


def ensemble_marker(func):
    """
    Decorator to mark a function or method with an attribute '__used_in_ensemble__'.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    setattr(wrapper, "__used_in_ensemble__", True)  # Set attribute here
    return wrapper


class MyClass:
    @ensemble_marker
    def my_method(self):
        print("This is a method within a class.")


my_obj = MyClass()
for name in dir(my_obj):
    member = getattr(my_obj, name)
    if is_desired_method(member) or is_desired_variable(name, member):
        print(f"Checking {name} {list(dir(member))}")
        if hasattr(member, "__used_in_ensemble__"):
            print(f"Adding {name} to ensemble")
