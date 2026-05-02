"""
Default experiment setup
"""
"""
%load_ext autoreload
%autoreload 1

%aimport quantdata
%aimport quantdata.biz_calendar
%aimport quantdata.widb
%aimport quantdata.widb.client
%aimport quantdata.widb.day
%aimport quantdata.widb.sampler
%aimport quantdata.dysanos
%aimport quantdata.utils.backbone
%aimport quantdata.torch
%aimport quantdata.torch.bbtorch
%aimport quantdata.torch.ffw
%aimport quantdata.dysanos.simple
%aimport quantdata.dysanos.simple.sampler
%aimport quantdata.dysanos.simple.model
%aimport quantdata.dysanos.simple.trainer
"""

import gc

from cdxcore.verbose import Context
verbose = Context("all")
with verbose.write_t("Importing torch...", end='') as tme:   
    import torch
    verbose.write(f"done; this took {tme}.", head=False)

from quantdata.dysanos.simple.sampler import SimpleSampler, SimpleConfig, SamplingConfig, ExpiryFilter, RawDayFilter, SampleDaysConfig, DateRange, PoolConfig
from datetime import date

def load_data(tickers : list[str] = ['SPX','NDX','RUT']):
    
    def test():
        simple_config = SimpleConfig(
                tickers = tickers,
                date_range = DateRange(start=date(2015, 1, 1), end=date(2025, 7, 1)),
                raw_day_filter = RawDayFilter(),
                expiry_filter = ExpiryFilter(),
                sample_days_config = SampleDaysConfig(),
                sampling_config = SamplingConfig(num_samples=20)
        )
        pool_config = PoolConfig(num_workers=4, threading=False) # no parallel processing for testing
        verbose = Context.all
        sampler = SimpleSampler(simple_config=simple_config, pool_config=pool_config, verbose=verbose)
        return sampler
    
    sampler = test()
    return sampler

import ipynbname
from quantdata.dysanos.simple.model import AEC, BaseEnvironment, FeedForward
from quantdata.dysanos.simple.trainer import Trainer, TrainConfig
from cdxcore.subdir import SubDir, Context

visualizer_config = AEC.VisualizerConfig(
    metrics=["excess_spread", "excess_spread_avg", "excess", "excess_avg"],
    details=True,
)  # set to None to turn off visualization

def run( label, *,
         version,
         aec_config, 
         sampler, 
         visualizer_config =visualizer_config,
         epochs = 1000, 
         batch_per_epoch = 10,
         trn_batch_size = 128,
         val_batch_size = 2048,
         debug = False ):

    print("Cleaning up memory... ", end='', flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("done.")

    env   = BaseEnvironment(device="cuda", debug=debug)
    model = AEC(env=env, 
                sampler=sampler,
                version=version,
                visualizer_config=visualizer_config,
                config=aec_config )
    print("model created.")

    train_config = TrainConfig(
        lr_init = 1E-3,
        lr_final = 1E-6,
        lr_init_steps = 25,
        trn_batch_size = trn_batch_size,
        val_batch_size = val_batch_size,
        batch_per_epoch = batch_per_epoch,
        experiment_group="v1"
    )

    verbose = Context("all")
    trainer = Trainer( label=label, model=model, sampler=sampler, config=train_config, verbose=verbose )
    _ = trainer(epochs=epochs)


