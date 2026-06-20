# external/ — upstream frameworks (pinned by SHA, installed editable)

These are cloned here and installed with `pip install -e` so any local edits take
effect immediately. The clones are gitignored (`external/*/`) — they are separate
repos with their own history, **not** part of hermes-quant. Each clone's pinned
commit is recorded in [`versions.lock`](versions.lock).

We **do not maintain forks** of these. Earlier they were forked under the
`Rhymer-Lcy` account, but those forks carried zero local commits, so they were
retired — clone straight from upstream and check out the pinned SHA. The locks
([`requirements.lock.txt`](../requirements.lock.txt) for vnpy,
[`requirements-rqalpha.lock.txt`](../requirements-rqalpha.lock.txt) for rqalpha)
already point at the upstream URLs + SHAs, so env rebuilds are reproducible without
any personal fork. **If you ever need to patch a framework's own source** (true
二开), fork it *then*, repoint that clone's `origin` to your fork, and rerun
`python scripts/lock_externals.py` to record it.

## On THIS PC (Windows — execution + backtest + data)

```bash
cd external
git clone https://github.com/vnpy/vnpy.git
git clone https://github.com/vnpy/vnpy_xt.git
git clone https://github.com/vnpy/vnpy_paperaccount.git
git clone https://github.com/ricequant/rqalpha.git
# check out the pinned commits (see versions.lock)
git -C vnpy              checkout 1b78494979deb4c4996f6b864f234d9839f2f239
git -C vnpy_xt           checkout 23a32da6d80cc14516fda4cc21fc4c6819a36c19
git -C vnpy_paperaccount checkout fcfe2b58965a0dc99b5cdbe075d5a372d8ef3ac2
git -C rqalpha           checkout 745d81cf11c7f620f1aec7f7b1447b26796f79be
```

Editable install into the `hermes` env (RQAlpha may need its own env due to
dependency pins — decide at install time):

```bash
conda activate hermes
pip install -e external/vnpy
pip install -e external/vnpy_paperaccount
pip install -e external/vnpy_xt
```

## Deferred ML-research repos (not cloned on this PC)

Clone these only if and when large-scale ML research is pursued, on whatever machine runs
the training:

```bash
git clone https://github.com/microsoft/qlib.git
git clone https://github.com/microsoft/RD-Agent.git
```

> Pin a specific Qlib git SHA — core Qlib has no tagged release since v0.9.7
> (Aug 2025) and active development has shifted to the RD-Agent ecosystem.
