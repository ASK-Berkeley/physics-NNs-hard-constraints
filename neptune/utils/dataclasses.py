from dataclasses import dataclass

# Wrapper for pytrees


def dataclass_wrapper(_cls=None,  *args, **kwargs):
    def wrap(cls):
        use_custom_init = '__init__' in cls.__dict__
        if use_custom_init:
            custom_init = cls.__init__
            delattr(cls, "__init__")
        else:
            return dataclass(cls, *args, **kwargs)
        cls = dataclass(cls, *args, **kwargs)
        setattr(cls, "__default_init__", custom_init)
        setattr(cls, "__init__", custom_init)
        return cls
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)
