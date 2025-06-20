# first line: 112
    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)
