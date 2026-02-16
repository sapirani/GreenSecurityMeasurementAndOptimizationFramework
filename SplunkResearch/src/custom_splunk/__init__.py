# Bridge __init__.py: redirect package resolution to the inner custom_splunk/ directory.
#
# When SplunkResearch/src is on sys.path (which happens when running scripts from
# that directory or via PYTHONPATH), Python finds this outer custom_splunk/ directory
# as a namespace package and shadows the pip-editable-installed inner package.
# This __init__.py fixes that by pointing __path__ at the inner directory so that
# sub-imports like ``custom_splunk.envs`` resolve correctly, then executes the
# inner __init__.py to trigger gymnasium environment registration.
import os as _os

# Redirect sub-package lookups (e.g. custom_splunk.envs) to the inner directory
__path__ = [_os.path.join(_os.path.dirname(__file__), 'custom_splunk')]

# Execute the inner __init__.py which registers gymnasium environments
_inner_init = _os.path.join(__path__[0], '__init__.py')
with open(_inner_init) as _f:
    exec(compile(_f.read(), _inner_init, 'exec'))
