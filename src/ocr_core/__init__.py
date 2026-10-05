from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("marathon-ocr-core")
except PackageNotFoundError:  # source checkout without installation
    __version__ = "0.0.0.dev0"
