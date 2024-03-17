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
"""tvm.contrib.msc.pipeline.dynamic"""

from typing import Tuple, Any, List

from tvm.contrib.msc.core.runtime import BaseJIT
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from .pipeline import BasePipeline
from .worker import MSCPipeWorker


class MSCDynamic(BasePipeline):
    """Dynamic of Pipeline, process dynamic model"""

    def setup(self) -> dict:
        """Setup the pipeline

        Returns
        -------
        info: dict
            The setup info.
        """

        self._jit, self._workers = None, {}
        self._current_stage, self._jit_caches = None, {}
        return super().setup()

    def _prepare(self, data_loader: Any) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Parameters
        ----------
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of prepare.
        report: dict
            The report of prepare.
        """

        hooks = {"pre_forward": [self.pre_forward], "post_forward": [self.post_forward]}
        if isinstance(self._model, dict) and "model" in self._model:
            worker_models = self._model["worker_models"]
            self._model, device, training = self.jit_cls.load_native(
                self._model["model"], self._config
            )
        else:
            worker_models = {}
            self._model, device, training = self.jit_cls.load_native(self._model, self._config)
        self._jit = self.jit_cls(
            self._model,
            inputs=[i[0] for i in self._config["inputs"]],
            outputs=self._config["outputs"],
            device=device,
            training=training,
            hooks=hooks,
            logger=self._logger,
        )
        self._jit.build()
        assert MSCStage.PREPARE in self._config["dataset"], "prepare dataset is needed"
        cnt, max_golden = 0, self._config["dataset"][MSCStage.PREPARE].get("max_golden", 5)
        self._current_stage, self._jit_caches = MSCStage.PREPARE, {}
        for inputs in data_loader():
            if cnt >= max_golden > 0:
                break
            self._jit.run(inputs)
            cnt += 1
        # create workers
        info, report = {}, {}
        for name, cache in self._jit_caches.items():
            runner_ctx, saver = self._jit.get_runner_ctx(name), cache.get("saver")
            assert saver, "Failed to record datas for " + name
            saver.finalize()

            def _to_input(i_name):
                i_info = saver.info["inputs"][i_name]
                return (i_name, i_info["shape"], i_info["dtype"])

            w_model = worker_models.get(name, runner_ctx["model"])
            w_config = msc_utils.copy_dict(self._config)
            w_config.update(
                {
                    "inputs": [_to_input(i) for i in saver.info["input_names"]],
                    "outputs": saver.info["output_names"],
                }
            )
            w_config["dataset"]["golden"] = {"loader": saver.folder}
            self._workers[name] = self.create_worker(w_model, name, w_config)
            with msc_utils.change_workspace(self._workspace.create_dir(name)):
                info[name], report[name] = self._workers[name].prepare()
        self._jit_caches = {}
        return info, report

    def _parse(self) -> Tuple[dict, dict]:
        """Parse relax module for the pipeline.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        info, report = {}, {}
        for name, worker in self._workers.items():
            with msc_utils.change_workspace(self._workspace.create_dir(name)):
                info[name], report[name] = worker.parse()
        return info, report

    def _tool_applied(self, tool_type: str) -> bool:
        """Check if the tool is applied

        Parameters
        ----------
        tool_type: str
            The tool type.

        Returns
        -------
        applied: bool
            Whether the tool is applied.
        """

        return all(w.tool_applied(tool_type) for w in self._workers.values())

    def _apply_tool(
        self, tool_type: str, knowledge: dict = None, data_loader: Any = None
    ) -> Tuple[dict, dict]:
        """Apply tool with runner

        Parameters
        ----------
        tool_type: str
            The tool type to apply.
        knowledge: dict
            The pre knowledge.
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of apply tool.
        report: dict
            The report of apply tool.
        """

        raise Exception("should support _apply_tool!!")

    def _create_runtime(
        self,
        stage: str,
        stage_config: dict = None,
        tools: List[str] = None,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> Tuple[dict, dict]:
        """Create runtime.

        Parameters
        ----------
        stage: str
            The pipeline stage.
        stage_config: dict
            The config of this stage.
        tools: list<str>
            The tools to apply.
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        info, report = {}, {}
        for name, worker in self._workers.items():
            with msc_utils.change_workspace(self._workspace.create_dir(name)):
                info[name], report[name] = worker.create_runner(
                    stage, stage_config, tools, visualize, profile, use_cache
                )
                self._jit.set_runner(name, worker.runner)
        return info, report

    def _export_model(self, stage: str, folder: msc_utils.MSCDirectory, dump: bool = True) -> Any:
        """Export the model

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        exported:
            The exported model.
        """

        model = self.jit_cls.dump_nativate(self._model, folder) if dump else self._model
        worker_models = {
            n: w.export_model(stage, folder.create_dir(n), dump) for n, w in self._workers.items()
        }
        return {"model": model, "worker_models": worker_models}

    def _destory(self):
        """Destory the pipeline"""

        for worker in self._workers.values():
            worker.destory()

    def get_runtime(self, ret_type: str = "runner") -> Any:
        """Get the runtime of pipeline

        Parameters
        ----------
        ret_type: str
            The return type runner| runnable| model.

        Returns
        -------
        runnable:
            The runnable object.
        """

        if ret_type == "runner":
            return self._jit
        if ret_type in ("model", "runnable"):
            return self._jit.jit_model
        raise TypeError("Unexpect return type " + str(ret_type))

    def pre_forward(self, runner_name: str, inputs: List[Tuple[str, Any]]) -> Any:
        """pre forward hook for jit model

        Parameters
        ----------
        runner_name: str
            The runner name.
        inputs:
            The msc format inputs.
        """

        cache = self._jit_caches.setdefault(runner_name, {})
        if self._current_stage == MSCStage.PREPARE:
            cache["inputs"] = inputs
        self._pre_forward(runner_name, inputs)

    def _pre_forward(self, runner_name: str, inputs: List[Tuple[str, Any]]) -> Any:
        """pre forward hook for jit model

        Parameters
        ----------
        runner_name: str
            The runner name.
        inputs:
            The msc format inputs.
        """

        return None

    def post_forward(
        self, runner_name: str, outputs: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """pre forward hook for jit model

        Parameters
        ----------
        runner_name: str
            The runner name.
        outputs:
            The outputs.

        Returns
        -------
        outputs:
            The outputs.
        """

        cache = self._jit_caches[runner_name]
        if self._current_stage == MSCStage.PREPARE:
            assert "inputs" in cache, "Failed to record inputs"
            if "saver" not in cache:
                golden = (
                    msc_utils.get_dataset_dir().create_dir(runner_name).relpath("Golden", False)
                )
                saver_options = {
                    "input_names": [i[0] for i in cache["inputs"]],
                    "output_names": [o[0] for o in outputs],
                }
                cache["saver"] = msc_utils.IODataSaver(golden, saver_options)
            cache["saver"].save_batch([i[1] for i in cache["inputs"]], [o[1] for o in outputs])
        return self._post_forward(runner_name, outputs)

    def _post_forward(
        self, runner_name: str, outputs: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """pre forward hook for jit model

        Parameters
        ----------
        runner_name: str
            The runner name.
        outputs:
            The outputs.

        Returns
        -------
        outputs:
            The outputs.
        """

        return outputs

    def _record_stage(self, stage: str, info: dict = None, report: dict = None):
        """Record the stage

        Parameters
        -------
        stage: str
            The compile stage
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        stage_report = {}
        for name, w_report in report.items():
            for k, v in w_report.items():
                stage_report.setdefault(k, {})[name] = v
        info = {k: v for k, v in info.items() if v}
        super()._record_stage(stage, info, stage_report)

    def pipe_mark(self, msg: Any) -> str:
        """Mark the message with pipeline info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "DYNAMIC " + str(msg)

    @property
    def jit_cls(self):
        return BaseJIT

    @property
    def worker_cls(self):
        return MSCPipeWorker


class TorchDynamic(MSCDynamic):
    """Dynamic of Pipeline, process torch dynamo"""

    @property
    def jit_cls(self):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.msc.framework.torch.runtime import TorchJIT

        return TorchJIT
