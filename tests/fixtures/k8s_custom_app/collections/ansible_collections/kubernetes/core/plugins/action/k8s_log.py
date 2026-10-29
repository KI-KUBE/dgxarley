"""Fake k8s_log: appends the call to the JSONL file named by K8S_CUSTOM_APP_RECORD instead of touching a cluster."""

import json
import os
from typing import Any

from ansible.plugins.action import ActionBase  # type: ignore[import-untyped]


class ActionModule(ActionBase):  # type: ignore[misc]
    def run(self, tmp: Any = None, task_vars: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = super().run(tmp, task_vars)
        with open(os.environ["K8S_CUSTOM_APP_RECORD"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"module": "k8s_log", "args": self._task.args}) + "\n")
        res.update(changed=False, log="", log_lines=[])
        return res
