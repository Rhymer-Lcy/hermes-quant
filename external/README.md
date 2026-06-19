# external/ — your forks (installed editable, not vendored)

These are cloned here and installed with `pip install -e` so secondary-dev edits
take effect immediately. The clones themselves are gitignored (`external/*/`) —
they are separate repos with their own history, not part of hermes-quant.

All forks are under your GitHub account **Rhymer-Lcy**. Clone the fork as
`origin` and add the upstream as `upstream` so you can pull updates.

## On THIS PC (Windows — execution + backtest + data)

```bash
cd external
gh repo clone Rhymer-Lcy/vnpy
gh repo clone Rhymer-Lcy/vnpy_xt
gh repo clone Rhymer-Lcy/vnpy_paperaccount
gh repo clone Rhymer-Lcy/rqalpha

# add upstreams for syncing
git -C vnpy             remote add upstream https://github.com/vnpy/vnpy.git
git -C vnpy_xt          remote add upstream https://github.com/vnpy/vnpy_xt.git
git -C vnpy_paperaccount remote add upstream https://github.com/vnpy/vnpy_paperaccount.git
git -C rqalpha          remote add upstream https://github.com/ricequant/rqalpha.git
```

Editable install into the `hermes` env (RQAlpha may need its own env due to
dependency pins — decide at install time):

```bash
conda activate hermes
pip install -e external/vnpy
pip install -e external/vnpy_paperaccount
pip install -e external/vnpy_xt
```

## On the CLUSTER (Linux — V100×8, ML research only)

Not cloned on this PC. On the cluster:

```bash
gh repo clone Rhymer-Lcy/qlib
gh repo clone Rhymer-Lcy/RD-Agent
git -C qlib    remote add upstream https://github.com/microsoft/qlib.git
git -C RD-Agent remote add upstream https://github.com/microsoft/RD-Agent.git
```

> Pin a specific Qlib git SHA — core Qlib has no tagged release since v0.9.7
> (Aug 2025) and active development has shifted to the RD-Agent ecosystem.
