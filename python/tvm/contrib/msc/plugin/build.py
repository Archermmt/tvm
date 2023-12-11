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
"""tvm.contrib.msc.plugin.build"""

from typing import List, Dict, Any, Optional
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.plugin.codegen import get_codegen
from .register import register_plugin


def build_plugins(
    plugins: Dict[str, dict],
    frameworks: List[str],
    install_dir: msc_utils.MSCDirectory,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    externs_dir: msc_utils.MSCDirectory = None,
    workspace: msc_utils.MSCDirectory = None,
):
    """Build the plugins

    Parameters
    ----------
    plugins: dict<str, dict>
        The plugins define.
    frameworks: list<str>
        The frameworks for plugin.
    install_dir: MSCDirectory:
        The install folder.
    codegen_config: dict<string, string>
        The config to generate code.
    print_config: dict<string, string>
        The config to print code.
    externs_dir: MSCDirectory
        The extern sources folder.
    workspace: MSCDirectory
        The workspace folder.
    """

    workspace = workspace or msc_utils.msc_dir(keep_history=False, cleanup=True)
    # register the plugins
    extern_sources, extern_libs = {}, {}
    for name, plugin in plugins.items():
        sources, libs = register_plugin(name, plugin, externs_dir)
        extern_sources.update(sources)
        extern_libs.update(libs)
    # build plugins for frameworks
    for framework in frameworks:
        codegen = get_codegen(
            framework,
            codegen_config,
            print_config,
            build_folder=workspace.create_dir(framework),
            output_folder=install_dir.create_dir(framework),
            extern_sources=extern_sources,
            extern_libs=extern_libs,
        )
        libs_file = codegen.build_libs()
        print("libs_file " + str(libs_file))
        raise Exception("stop here!!")
        if codegen.need_manager:
            files = codegen.build_manager()
            print("manager files " + str(files))
        if codegen.need_convert:
            files = codegen.build_convert()
            print("convert files " + str(files))
        raise Exception("stop here!!")


def build_plugins_manager(
    plugins: Dict[str, dict],
    frameworks: List[str],
    install_dir: msc_utils.MSCDirectory,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    externs_dir: msc_utils.MSCDirectory = None,
    workspace: msc_utils.MSCDirectory = None,
) -> Dict[str, Any]:
    """Build the plugins and load plugin manager

    Parameters
    ----------
    plugins: dict<str, dict>
        The plugins define.
    frameworks: list<str>
        The frameworks for plugin.
    install_dir: MSCDirectory:
        The install folder.
    codegen_config: dict<string, string>
        The config to generate code.
    print_config: dict<string, string>
        The config to print code.
    externs_dir: MSCDirectory
        The extern sources folder.
    workspace: MSCDirectory
        The workspace folder.

    Returns
    -------
    managers: dict<str, PluginManager>
        The plugin managers for each framework.
    """

    install_dir = build_plugins(
        plugins, frameworks, install_dir, codegen_config, print_config, externs_dir, workspace
    )


def build_plugins_wheel(
    project_name: str,
    plugins: Dict[str, dict],
    frameworks: List[str],
    install_dir: msc_utils.MSCDirectory,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    externs_dir: msc_utils.MSCDirectory = None,
    workspace: msc_utils.MSCDirectory = None,
) -> str:
    """Build the plugins and build to wheel

    Parameters
    ----------
    project_name: str
        The project name
    plugins: dict<str, dict>
        The plugins define.
    frameworks: list<str>
        The frameworks for plugin.
    install_dir: MSCDirectory:
        The install folder.
    codegen_config: dict<string, string>
        The config to generate code.
    print_config: dict<string, string>
        The config to print code.
    externs_dir: MSCDirectory
        The extern sources folder.
    workspace: MSCDirectory
        The workspace folder.

    Returns
    -------
    wheel_path: str
        The file path of wheel.
    """

    install_dir = build_plugins(
        plugins, frameworks, install_dir, codegen_config, print_config, externs_dir, workspace
    )
