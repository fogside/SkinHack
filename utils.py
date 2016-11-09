def _start_shell(local_ns=None):
    """Starts interactive ipython shell within local context
        use as `_start_shell(locals())`
    """
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)
