import numpy as np
from matplotlib import pyplot as plt
from neuron import h

from model.morphology import MultiDend
from model.nrn_simulation import get_base_vm_cli, hoc_run
from utils import settings
from utils.plot_utils import annotate_cols_rows, use_scalebar
from shared import INIT

INIT()

def run_clustered_vs_diffused(
    cli=5,
    total_synapses=24,
    multi_dend_kwargs=None,
    hz=10,
    duration=2000,
    weight=10,
    with_kcc2=False,
    recover=0,
    quick=True,
    nrn_graphing=False,
):
    total_time = duration
    if recover > 0:
        total_time += recover
    if multi_dend_kwargs is None:
        multi_dend_kwargs = dict(num_dendrites=1, soma=True, axon=False)
    # create neuron 1 (diffused_neuron synapses)
    diffused_neuron = MultiDend("diffused", **multi_dend_kwargs)
    if with_kcc2:
        diffused_neuron.add_kcc2()
    diffused_neuron.set_cl(cli=cli)
    all_synapses = diffused_neuron.add_n_synapses(
        "GABAa", n=total_synapses, sections=diffused_neuron.radial_dends, gmax=0.0005
    )

    # create neuron 2 (clustered_neuron synapses)
    clustered_neuron = MultiDend("clustered", **multi_dend_kwargs)
    if with_kcc2:
        clustered_neuron.add_kcc2()
    clustered_neuron.set_cl(cli=cli)
    proximal_synapses = clustered_neuron.add_n_synapses(
        "GABAa",
        n=total_synapses,
        sections=clustered_neuron.radial_dends[0],
        gmax=0.0005,
    )

    # stimulation
    start = 0
    diffused_neuron.netstim(
        hz=hz, start=start, duration=duration, weight=weight, synapses=all_synapses
    )
    clustered_neuron.netstim(
        hz=hz, start=start, duration=duration, weight=weight, synapses=proximal_synapses
    )

    # graphs
    if settings.NEURON_GUI and nrn_graphing:
        g_diffused = h.Graph(0)
        #        label      var    col brush    section
        g_diffused.addvar(
            "diffused_neuron soma.v", "v(0.5)", 2, 2, sec=diffused_neuron.soma
        )
        g_diffused.addvar("soma.ecl", "ecl(0.5)", 2, 2, sec=diffused_neuron.soma)
        g_diffused.addvar(
            "prox0.v", "v(0.5)", 3, 1, sec=diffused_neuron.radial_dends[0]
        )
        g_diffused.addvar(
            "prox0.ecl", "ecl(0.5)", 3, 1, sec=diffused_neuron.radial_dends[0]
        )

        g_clustered = h.Graph(0)
        g_clustered.addvar(
            "clustered_neuron soma.v", "v(0.5)", 2, 2, sec=clustered_neuron.soma
        )
        g_clustered.addvar("soma.ecl", "ecl(0.5)", 2, 2, sec=clustered_neuron.soma)
        g_clustered.addvar(
            "prox0.v", "v(0.5)", 3, 1, sec=clustered_neuron.radial_dends[0]
        )
        g_clustered.addvar(
            "prox0.ecl", "ecl(0.5)", 3, 1, sec=clustered_neuron.radial_dends[0]
        )

        g_diffused.view(-70, -95, h.tstop + 100, 25, 1200, 1000, 1000, 400)
        g_clustered.view(-70, -95, h.tstop + 100, 25, 1200, 1000, 1000, 400)

        if g_diffused not in h.graphList[0]:
            h.graphList[0].append(g_diffused)
        if g_clustered not in h.graphList[0]:
            h.graphList[0].append(g_clustered)

    record_loc = {"soma": 0.5}
    hoc_run(
        tstop=total_time,
        quick=quick,
        scale=nrn_graphing,
        record_from=[diffused_neuron, clustered_neuron],
        record_args=[
            {"record_var": "v", "locations": record_loc},
            {"record_var": "cli", "locations": record_loc},
        ],
    )

    return diffused_neuron, clustered_neuron


def create_fig_clustered_vs_diffused():
    multi_dend_kwargs = dict(num_dendrites=4, even_radial=False, soma=True, axon=False)
    vm, base_cli = get_base_vm_cli(neuron_model=MultiDend, **multi_dend_kwargs)

    kwargs = dict(
        total_synapses=24,
        multi_dend_kwargs=multi_dend_kwargs,
        hz=10,
        duration=10000,
        weight=10,
        recover=0,
        quick=True,
        nrn_graphing=True,
    )

    h.v_init = vm
    diffused, clustered = run_clustered_vs_diffused(cli=base_cli, **kwargs)
    diffused_with_kcc2, clustered_with_kcc2 = run_clustered_vs_diffused(
        cli=base_cli, with_kcc2=True, **kwargs
    )
    
    # plot        |        vm               |    cli
    #   w/ kcc2   | [clustered, diffused]   |
    #   w/o kcc2  |                         |
    fig, axes = plt.subplots(2, 2, sharey="col", sharex="all")
    alpha = 0.5
    diffused.plot(axes[0, :], alpha=alpha)
    clustered.plot(axes[0, :], alpha=alpha)
    diffused_with_kcc2.plot(axes[1, :],  alpha=alpha)
    clustered_with_kcc2.plot(axes[1, :], alpha=alpha)

    annotate_cols_rows(
        axes,
        cols=["Membrane Voltage", "Chloride Concentration"],
        rows=[settings.STATIC_CHLORIDE_STR_LONG, settings.DYNAMIC_CHLORIDE_STR_LONG],
    )
    axes[0, 0].legend(["Diffused Synapses", r"Clustered Synapses"], loc=2)
    
    # ADD SCALEBAR
    # if we want a single scalebar to represent both graphs, need to have the ticks the same
    yticks_0 = axes[0, 0].get_yticks()  # mV
    yticks_1 = axes[0, 1].get_yticks()  # mM
    if len(yticks_1) > len(yticks_0):
        yticks_0 = np.linspace(yticks_0[0], yticks_0[-1], len(yticks_1))
    elif len(yticks_1) < len(yticks_0):
        yticks_1 = np.linspace(yticks_1[0], yticks_1[-1], len(yticks_0))
    axes[0, 0].set_yticks(yticks_0)
    axes[1, 0].set_yticks(yticks_0)
    axes[0, 1].set_yticks(yticks_1)
    axes[1, 1].set_yticks(yticks_1)

    loc_args = dict(
        loc="upper right",
        bbox_to_anchor=(0.2, 1.1, 0.0, 0),
        bbox_transform=axes[1, 1].transAxes,
    )
    use_scalebar(axes[0, 0], labelx="ms", labely="mV", labelypad=-2, **loc_args)
    use_scalebar(axes[0, 1], labelx="ms", labely=settings.mM, labelypad=0, **loc_args)
    for ax in axes[1, :]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)


if __name__ == "__main__":
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid", palette="Set2", rc={"lines.linewidth": 1})
    create_fig_clustered_vs_diffused()
    plt.show()