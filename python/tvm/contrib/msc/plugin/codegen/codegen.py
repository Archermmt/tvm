# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.core.codegen.codegen"""

import os
import subprocess
from typing import Dict, List, Optional

from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.plugin import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .sources import get_plugin_sources


class BasePluginCodeGen(object):
    """Manager class to generate codes and build plugin

    Parameters
    ----------
    codegen_config: dict<string, string>
        The config to generate code.
    print_config: dict<string, string>
        The config to print code.
    build_folder: MSCDirectory
        The codegen folder.
    output_folder: MSCDirectory
        The output folder.
    extern_sources: dict<string, string>
        The depend source files.
    extern_libs: dict<string, string>
        The depend lib files.
    """

    def __init__(
        self,
        codegen_config: Optional[Dict[str, str]] = None,
        print_config: Optional[Dict[str, str]] = None,
        build_folder: msc_utils.MSCDirectory = None,
        output_folder: msc_utils.MSCDirectory = None,
        extern_sources: Dict[str, str] = None,
        extern_libs: Dict[str, str] = None,
    ):
        self._codegen_config = msc_utils.copy_dict(codegen_config)
        self._print_config = msc_utils.copy_dict(print_config)
        self._build_folder = build_folder or msc_utils.msc_dir(keep_history=False, cleanup=True)
        self._output_folder = output_folder or msc_utils.msc_dir("msc_plugins")
        self._extern_sources = extern_sources or {}
        self._extern_libs = extern_libs or {}
        self.setup()

    def setup(self):
        """Set up the codegen"""

        self._codegen_config = msc_utils.dump_dict(self._codegen_config)
        self._print_config = msc_utils.dump_dict(self._print_config)

    def build_libs(self) -> List[str]:
        """Generate source and build the lib

        Returns
        -------
        paths: list<str>
            The lib file paths.
        """

        sources = self.source_getter(self._codegen_config, self._print_config, "sources")
        lib_dir, lib_files = self._output_folder.create_dir("libs"), []
        with self._build_folder as folder:
            # add depends
            with folder.create_dir("src") as src_folder:
                for name, file in self._extern_sources.items():
                    src_folder.copy(file, name)
                with src_folder.create_dir("utils") as utils_folder:
                    for name, source in get_plugin_sources().items():
                        utils_folder.add_file(name, source)
            for name, source in sources.items():
                if name == "CMakeLists.txt":
                    folder.add_file(name, source)
                else:
                    folder.add_file(os.path.join("src", name), source)
            with folder.create_dir("build") as build_folder:
                command = "cmake ../ && make"
                with open("codegen.log", "w") as log_f:
                    process = subprocess.Popen(command, stdout=log_f, stderr=log_f, shell=True)
                process.wait()
                assert (
                    process.returncode == 0
                ), "Failed to build plugin under {}, check codegen.log for detail".format(
                    os.getcwd()
                )
                for f in build_folder.listdir():
                    if not f.endswith(".so"):
                        continue
                    lib_files.append(folder.copy(f, lib_dir.relpath(f)))
        return lib_files

    def build_manager(self) -> List[str]:
        """Generate manager source for plugin

        Returns
        -------
        paths: list<str>
            The manager file paths.
        """

        sources = self.source_getter(self._codegen_config, self._print_config, "manager")
        manager_files = []
        with self._output_folder as folder:
            for name, source in sources.items():
                manager_files.append(folder.add_file(name, source))
        return manager_files

    def build_convert(self) -> List[str]:
        """Generate manager source for plugin

        Returns
        -------
        paths: list<str>
            The convert file paths.
        """

        sources = self.source_getter(self._codegen_config, self._print_config, "convert")
        convert_files = []
        with self._output_folder as folder:
            for name, source in sources.items():
                convert_files.append(folder.add_file(name, source))
        return convert_files

    @property
    def source_getter(self):
        raise NotImplementedError("source_getter is not supported for Base codegen")

    @property
    def need_manager(self):
        return True

    @property
    def need_convert(self):
        return True


class TVMPluginCodegen(BasePluginCodeGen):
    @property
    def source_getter(self):
        return _ffi_api.GetTVMPluginSources

    @property
    def need_convert(self):
        return False


class TorchPluginCodegen(BasePluginCodeGen):
    def setup(self):
        """Set up the codegen"""

        import torch.utils

        self._codegen_config["torch_prefix"] = torch.utils.cmake_prefix_path
        super().setup()

    @property
    def source_getter(self):
        return _ffi_api.GetTorchPluginSources


class TensorRTPluginCodegen(BasePluginCodeGen):
    @property
    def source_getter(self):
        return _ffi_api.GetTensorRTPluginSources

    @property
    def need_convert(self):
        return False


def get_codegen(
    framework: str,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    output_folder: msc_utils.MSCDirectory = None,
    extern_sources: Dict[str, str] = None,
    extern_libs: Dict[str, str] = None,
):
    """Create codegen for framework

    Parameters
    ----------
    framework: str
        THe framework for the plugin.
    codegen_config: dict<string, string>
        The config to generate code.
    print_config: dict<string, string>
        The config to print code.
    build_folder: MSCDirectory
        The codegen folder.
    output_folder: MSCDirectory
        The output folder.
    extern_sources: dict<string, string>
        The depend source files.
    extern_libs: dict<string, string>
        The depend lib files.
    """

    if framework == MSCFramework.TVM:
        return TVMPluginCodegen(
            codegen_config, print_config, build_folder, output_folder, extern_sources, extern_libs
        )
    if framework == MSCFramework.TORCH:
        return TorchPluginCodegen(
            codegen_config, print_config, build_folder, output_folder, extern_sources, extern_libs
        )
    if framework == MSCFramework.TENSORRT:
        return TensorRTPluginCodegen(
            codegen_config, print_config, build_folder, output_folder, extern_sources, extern_libs
        )
    raise NotImplementedError("framework {} is not support for plugin codegen".format(framework))
