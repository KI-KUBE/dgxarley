"""Helper scripts for operating the K3s cluster.

Contains three standalone CLIs:

* :mod:`dgxarley.k3shelperstuff.update_local_k3s_keys` -- keep the local
  ``~/.kube/config`` in sync with the kubeconfig of a remote K3s server.
* :mod:`dgxarley.k3shelperstuff.keel_drift` -- find Keel-tracked workloads whose
  running image lags behind the image their tag currently points at.
* :mod:`dgxarley.k3shelperstuff.pin_drift` -- find version pins in the repo that
  lag behind the upstream release.

``keel_drift`` needs the optional ``dgxarley[k3s]`` dependencies (``kubernetes``,
``typer``), ``pin_drift`` only ``typer`` out of them; ``update_local_k3s_keys``
runs on the standard library alone.
Importing this package itself pulls in nothing extra.
"""
