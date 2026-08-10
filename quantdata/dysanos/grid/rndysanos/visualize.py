"""Run with ``marimo run quantdata/dysanos/grid/rndysanos/visualize.py``."""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from quantdata.dysanos.grid.rndysanos.visualization import (
        discover_checkpoints,
        load_checkpoint_history,
        plot_checkpoint,
    )

    return discover_checkpoints, load_checkpoint_history, mo, plot_checkpoint


@app.cell
def _(mo):
    root = mo.ui.text(value="~/.cache/rndysanos/oce", label="Checkpoint root")
    group = mo.ui.text(value="", label="Group/session")
    refresh = mo.ui.run_button(label="Refresh")
    mo.hstack([root, group, refresh])
    return group, refresh, root


@app.cell
def _(discover_checkpoints, group, mo, refresh, root):
    refresh
    paths = discover_checkpoints(root.value, group.value)
    choices = {path.name: str(path) for path in paths}
    selected = mo.ui.dropdown(
        options=choices,
        value=next(iter(choices), None),
        label=f"Checkpoint ({len(paths)} found)",
    )
    selected
    return (selected,)


@app.cell
def _(load_checkpoint_history, mo, plot_checkpoint, selected):
    if selected.value is None:
        output = mo.callout("No checkpoints found for this root/group.", kind="warn")
    else:
        state = load_checkpoint_history(selected.value)
        figure = plot_checkpoint(state)
        output = mo.vstack(
            [
                mo.md(
                    f"**Phase:** `{state['phase']}` · **main epoch:** "
                    f"{state['main_epoch']} · **moment epoch:** {state['moment_epoch']}"
                ),
                figure,
            ]
        )
    output
    return


if __name__ == "__main__":
    app.run()
