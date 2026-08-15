import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra = [] if sys.platform.startswith("win") else ["-O3"]

setup(
    name="pora-cpp",
    version="0.1.0",
    description="C++ comparison implementation of the pora_rs hot loop",
    ext_modules=[
        Pybind11Extension(
            "pora_cpp", ["pora_cpp.cpp"], cxx_std=17, extra_compile_args=extra
        )
    ],
    cmdclass={"build_ext": build_ext},
)
