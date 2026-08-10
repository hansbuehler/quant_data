"""
Whole training setup
"""

import gc
from dataclasses import dataclass
from datetime import date

import numpy as np
from cdxcore.verbose import Context
from prettytable import PrettyTable


@dataclass(frozen=True)
class RNDysanosExperiment:
    grid_strikes : int = 100   #: The number of strikes per expiry we are matching the market with
    num_nms      : int = 20    #: The nunber of strikes per expiry we are simulating
    num_alpha    : int|None = 5
    use_matrix_exp : bool = False
    bandwidth    : float = 0.2
    num_steps    : int = 10
    num_samples  : int = 100_000
    pcnt_validation : float = 0.1
    clip_grad_norm : float|None = None
    cnt_universe : int = 1
    universe_seed_offset : int = 0

    def __str__(self):
        return f"{self.num_samples},steps={self.num_steps},nms={self.num_nms},gs={self.grid_strikes},uni={self.cnt_universe},seed_offset={self.universe_seed_offset}.alpha={self.num_alpha},matrix_exp={self.use_matrix_exp},bandwidth={self.bandwidth},pcnt_validation={self.pcnt_validation},clip_grad_norm={self.clip_grad_norm}"

# ``create_universe`` below already caches each explicit RNG stream.  Do not
# cache this aggregate function: a ten-stage experiment would duplicate every
# source universe in a second multi-universe HDF5 file, and parallel regimes
# would race to write that tens-of-gigabytes aggregate.
def prep_experiment( experiment : RNDysanosExperiment, *,  plot_sampler : bool = True ):
    gc.collect()
    verbose = Context.all
    with verbose.write_t("Importing torch...", end='') as tme:
        import torch
        del torch
        verbose.write(f"done; this took {tme}.", head=False)

    # load market data
    # ----------------

    from quantdata.dysanos.grid.torch_market import (
        DateRange,
        GridConfig,
        GridMarketDefinition,
        PoolConfig,
    )
    gc.collect()

    verbose.write("Starting to load market data...")
    grid_config     = GridConfig(grid_num_strikes=experiment.grid_strikes)
    grid_market_def = GridMarketDefinition( ticker="SPX", date_range=DateRange(start=date(2020, 1, 1), end=date(2025, 8, 30)), grid_config=grid_config )
    market          = grid_market_def.load( pool_config = PoolConfig(num_workers=4, threading=False ), override_cache_mode="cacheonly", verbose=verbose(1) )
    verbose.write(f"Loaded market data: found {len(market['dates'])} dates.")

    # load trained model
    # ------------------

    from quantdata.dysanos.grid.torch_gfx import VizConfig
    from quantdata.dysanos.grid.torch_model import (
        BaseEnvironment,
        FeedForward,
        SanosDecoderConfig,
        SanosModelConfig,
    )
    from quantdata.dysanos.grid.torch_train import TrainConfig, train
    from quantdata.pool import PoolConfig

    mdl_config = SanosModelConfig( 
        label="ac_state20",
        user_version="0.2",
        group="sm25_100",
        encoder_config = FeedForward.Config(
                activation = "SELU",
                num_depth = 2,
                num_width = 100,
                use_bias = True,
            ),
        decoder_config = SanosDecoderConfig(
            smoothness = 0.25,
            num_state = 20, 
            num_nms = experiment.num_nms,
            nn_config = FeedForward.Config(
                activation = "SELU",
                num_depth = 2,
                num_width = 100,
                use_bias = True,
            ),
        ),
    )

    model        = None
    torch_data   = None
    env          = BaseEnvironment(device="cpu", debug=False)
    viz_config   = VizConfig( num_viz_dates=4 )
    trn_config   = TrainConfig( num_epochs=100_000, lr=1E-4, final_lr=1E-6, weight_decay=1e-4, batch_size=400, compile=True, epoch_interval=10000, clip_grad_norm=experiment.clip_grad_norm )
    with verbose.write_t("Loading trained model...", end='') as tme:
        model_output,\
        model,\
        torch_data   = train( market, mdl_config=mdl_config, trn_config=trn_config, viz_config=viz_config, just_load="plot", env=env )
        torch_data   = mdl_config.torch_data( env = env, market = market, verbose = Context.quiet ) if torch_data is None else torch_data
        del model_output, mdl_config, viz_config, env
        gc.collect()
        verbose.write("loaded; this took {tme}.", head=False)

    # AR(1) PCA
    # --------=-

    from quantdata.dysanos.grid.torch_simulator import ARSampler, ARSamplerConfig
    from quantdata.dysanos.grid.torch_simulator import plot_sampler as plot_ar_sampler

    verbose.write("Computing AR1(1) PCA")
    ar_config      = ARSamplerConfig(
        num_alpha=experiment.num_alpha,
        use_matrix_exp=experiment.use_matrix_exp,
    )
    sampler        = ARSampler( market, model, torch_data, config=ar_config, verbose=verbose(1) )
    gc.collect()
    verbose.write("AR(1) PCA computed:")
    sampler.report_system()

    if plot_sampler:
        plot_ar_sampler(sampler)

    # simulate universe
    # -----------------

    from quantdata.dysanos.grid.torch_universe import create_universe

    universes = []

    for i in range(experiment.cnt_universe):
        verbose.write(f"Simulating universe {i+1}/{experiment.cnt_universe}")

        # A cache hit in create_universe does not advance sampler.rng.  Assign
        # an explicit per-universe stream so every cache key and path sample is
        # distinct even when all preceding universes came from cache.
        sampler.rng = np.random.default_rng(
            seed=sampler.config.seed + experiment.universe_seed_offset + i
        )

        S    = experiment.num_samples
        TAU  = experiment.num_steps-1
        T,_  = grid_config.TN
        N    = experiment.num_nms
        XS   = sampler.num_xstate

        universe = create_universe(
                                model=model, sampler=sampler, market=market, reset_rng=False,
                                num_steps=experiment.num_steps,
                                num_samples=experiment.num_samples,
                                block_size=100000,
                                bandwidth=experiment.bandwidth,
                                verbose=verbose(1))
        print(f"Found universe['C_tp1'].shape={universe['C_tp1'].shape}", flush=True)
        # reduce
        universe = dict(
            dC = universe["C_tp1"]-universe["C_t"],
            dS = universe["spots"][:,1:]-universe["spots"][:,:-1],
            C = universe["C_t"],
            spot = universe["spots"][:,:-1],
            delta = universe["delta_t"],
            expiries = universe["expiries"][None,None,:].astype(np.float32),
            log_strikes = np.log( universe["strikes_t"] ),
            xstate = universe['xstate'].astype(np.float32),
        )
        assert universe["dC"].shape == (S, TAU, T, N), f"universe['dC'].shape={universe['dC'].shape} != {(S, TAU, T, N)}"
        assert universe["C"].shape == (S, TAU, T, N), f"universe['C'].shape={universe['C'].shape} != {(S, TAU, T, N)}"
        assert universe["delta"].shape == (S, TAU, T, N), f"universe['delta'].shape={universe['delta'].shape} != {(S, TAU, T, N)}"
        assert universe["dS"].shape == (S, TAU,), f"universe['dS'].shape={universe['dS'].shape} != {(S, TAU, )}"
        assert universe["spot"].shape == (S, TAU,), f"universe['spot'].shape={universe['spot'].shape} != {(S, TAU, )}"
        assert universe["expiries"].shape == (1, 1, T), f"universe['expiries'].shape={universe['expiries'].shape} != {(1, 1, T)}"
        assert universe["log_strikes"].shape == (S, TAU, T, N), f"universe['log_strikes'].shape={universe['log_strikes'].shape} != {(S, TAU, T, N)}"
        xstate = universe['xstate'][:,:-1,...]
        assert xstate.shape == (S, TAU, XS), f"xstate.shape={xstate.shape} != {(S, TAU, XS)}"

        fields = sorted(universe)
        tbl = PrettyTable(field_names=["field", "shape", "nbytes", "dtype"])
        for k in fields:
            v = universe[k]
            tbl.add_row([k, v.shape, v.nbytes, v.dtype])
        verbose.write(f"Simulated universe {i+1}/{experiment.cnt_universe}")
        verbose.report(1,tbl)
        universes.append(universe)
    gc.collect()
    return universes


