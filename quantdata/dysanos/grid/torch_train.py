from dataclasses import dataclass, field, asdict
import math
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir
from cdxcore.err import verify, verify_inp
from cdxcore.version import version
from cdxcore.uniquehash import unique_label48_8
from quantdata.dysanos.grid.torch_market import GridMarketDefinition
from quantdata.dysanos.grid.torch_model import SanosModelConfig, BaseEnvironment
from quantdata.dysanos.grid.torch_gfx import VizConfig, Visualizer
from quantdata.pool import PoolConfig
import torch as torch
import time as time
import numpy as np
import os
import warnings

if os.environ.get("DYSANONS_USE_LAPTOP") is not None:
    _cache_root = SubDir(r"C:\users\public\.dysanos_grid/")
else:
    _cache_root = SubDir("~/.dysanos_grid")
    
@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    num_epochs          : int = 5000       #: total epochs
    batch_size          : int|None = None  #: batch size, or None for all samples
    val_batch_size      : int|None = None  #: batch size for validation, or None for all samples
    pcnt_validation     : float = 0.       #: fraction of samples to use for validation. Will be automatically adjusted such that there are always at least 2 training samples.
    lr                  : float = 1E-3     #: initial learning rate
    final_lr            : float = 1E-5     #: final learning rate
    weight_decay        : float = 1e-4     #: weight decay for regularization
    compile             : bool = False     #: compile torch
    clip_grad_norm      : float|None = None #: max norm for gradient clipping
    clip_grad           : float|None = None #: max for gradient clipping

    epoch_interval: int = 200         #: after how many epochs to update visualization and store checkpoint
    time_interval : float = 2*60      #: after how many seconds to update visualization and store checkpoint

    def __post_init__(self):
        verify_inp( self.num_epochs > 0, f"Expected 'num_epochs' to be positive, but found {self.num_epochs}" )
        verify_inp( self.final_lr > 0., f"Expected 'final_lr' to be positive, but found {self.final_lr}" )
        verify_inp( self.lr >= self.final_lr, f"Expected 'lr' to be greater than or equal to 'final_lr' {self.final_lr}, but found {self.lr}" )
        verify_inp( self.weight_decay >= 0., f"Expected 'weight_decay' to be non-negative, but found {self.weight_decay}" )
        verify_inp( self.epoch_interval > 0, f"Expected 'epoch_interval' to be positive, but found {self.epoch_interval}" )
        verify_inp( self.time_interval > 0., f"Expected 'time_interval' to be positive, but found {self.time_interval}" )
        verify_inp( self.pcnt_validation >= 0. and self.pcnt_validation < 1., f"Expected 'pcnt_validation' to be in [0,1), but found {self.pcnt_validation:.2f}" )
        verify_inp( self.weight_decay >= 0., f"Expected 'weight_decay' to be non-negative, but found {self.weight_decay}" )
        verify_inp( self.clip_grad_norm >= 0., f"Expected 'clip_grad_norm' to be non-negative, but found {self.clip_grad_norm}" ) if not self.clip_grad_norm  is None else None
        verify_inp( self.clip_grad >= 0., f"Expected 'clip_grad' to be non-negative, but found {self.clip_grad}" ) if not self.clip_grad  is None else None

    def __str__(self):
        return f"epochs={self.num_epochs},lr={self.lr:.4g}->{self.final_lr:.4g},wdec={self.weight_decay:.4g},bsize={'all' if self.batch_size is None else self.batch_size},vbsize={'all' if self.val_batch_size is None else self.val_batch_size},"+\
                f"pcnt_val={self.pcnt_validation:.2f},compile={self.compile},epoch_interval={self.epoch_interval},time_interval={self.time_interval:.1f}s"+\
                f"clip={self.clip_grad_norm:.2g},{self.clip_grad:.2g}"

    def lr_for_epoch(self, epoch: int) -> float:
        if epoch <= 0:
            return self.lr
        if epoch >= self.num_epochs - 1:
            return self.final_lr
        p = epoch / (self.num_epochs - 1)
        return self.lr * (self.final_lr / self.lr) ** p

    def set_lr_for_epoch(self, epoch: int, optimizer: torch.optim.Optimizer) -> float:
        lr = self.lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_trn_vld_samples(self, num_total_samples : int) -> tuple[int,int]:
        num_trn_samples   = max(2, min(math.ceil(num_total_samples * (1. - self.pcnt_validation)), num_total_samples))
        num_vld_samples   = num_total_samples - num_trn_samples
        return num_trn_samples, num_vld_samples

@dataclass
class MetricTracker:
    trn_epochs : list[int] = field(default_factory=list)
    trn_losses : list[float] = field(default_factory=list)
    lr : list[float] = field(default_factory=list)
    val_epochs : list[int] = field(default_factory=list)
    val_losses : list[float] = field(default_factory=list)
    vld_epochs : list[int] = field(default_factory=list)
    vld_losses : list[float] = field(default_factory=list)

def train( market        : PrettyObject,*,
           mdl_config    : SanosModelConfig, 
           trn_config    : TrainConfig = TrainConfig(),
           viz_config    : VizConfig = VizConfig(),
           restart       : bool = False,
           just_load     : bool = False,
           env           : BaseEnvironment = BaseEnvironment(),
           verbose       : Context = Context.all ) -> dict:
    """
    Train model
    """
    verify_inp( not (restart and just_load), f"Expected at most one of 'restart' and 'just_load' to be True, but found both True" )

    verbose = verbose if not verbose is None else Context.all
    verbose = verbose if not just_load else Context.quiet

    verbose.write(lambda : f"Training {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' for {trn_config.num_epochs} epochs on {market.grid_market_def.ticker} for date range {market.grid_market_def.date_range}")
    verbose.write(lambda : f"Using {'all samples' if trn_config.batch_size is None else f'batches of {trn_config.batch_size} samples'} with learning rate {trn_config.lr:.4g} and weight decay {trn_config.weight_decay:.4g}.")
    verbose.report(1,lambda : f"Model config: {mdl_config}")
    verbose.report(1,lambda : f"Grid config: {market.grid_market_def.grid_config}")
    verbose.report(1,lambda : f"Mk Arb Free pre-processor: {market.grid_market_def.mk_arb_free}")
    verbose.report(1,lambda : f"Options filter: {market.grid_market_def.options_filter}")
    verbose.report(1,lambda : f"Raw day filter: {market.grid_market_def.raw_day_filter}")
    verbose.report(1,lambda : f"Expiry filter: {market.grid_market_def.expiry_filter}")

    num_total_samples = len(market.dates)
    num_trn_samples, num_vld_samples = trn_config.get_trn_vld_samples(num_total_samples)
    verify_inp( num_vld_samples == 0, f"Cannot train with validation samples")

    # create model
    # ------------
    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        T, N      = market.grid_market_def.TN
        model     = mdl_config.create(  env,
                                        num_expiries=T,
                                        num_market_strikes=N )
        verbose.write(lambda : f"done; this took {tme}.", head=False )
        if model.debug:
            warnings.warn("** WARNING: model is running in (very slow) debug mode. **")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trn_config.lr,
        weight_decay=trn_config.weight_decay
    )

    # load
    # ----

    path          = _cache_root(ext="pt") if mdl_config.group == "" else _cache_root(mdl_config.group,ext="pt")
    file_name     = unique_label48_8(mdl_config.label, as_file_name=True)
    cache_ffn     = path.full_file_name(file_name)
    start_epoch   = 0
    tracker       = MetricTracker()
    model_version = model.version.unique_id48

    if not restart and path.exists(file_name):
        verbose.write(lambda : f"Found existing recovery checkpoint '{cache_ffn}'. Restoring model state from this checkpoint.")

        state = torch.load(cache_ffn, map_location=env.device, weights_only=False)
        user_version = state.pop("version", None)
        mdl_version  = state.pop("mdl_version", None)
        err          = None
        if user_version is None:
            err = f"Checkpoint '{cache_ffn}' does not contain a 'version'. This is likely an invalid checkpoint file."
        elif user_version != mdl_config.user_version:
            err = f"Checkpoint '{cache_ffn}' file user_version {user_version} does not match model user_version {mdl_config.user_version}."
        elif not mdl_version is None and model_version != mdl_version:
            err = f"Checkpoint '{cache_ffn}' file model version {mdl_version} does not match model version {model_version}."
        else:
            model.load_state_dict(state.pop("model_state_dict"))
            if just_load:
                model_output = None
                tdata        = None
                return model_output,  model, tdata

            optimizer.load_state_dict(state.pop("optimizer_state_dict"))
            tracker = MetricTracker(**state.pop("tracker"))
            start_epoch = tracker.trn_epochs[-1] + 1

        if not err is None:
            if just_load:
                raise RuntimeError(err)
            warnings.warn(err + f" -- Restarting training.")

    else:
        if just_load:
            if restart:
                raise RuntimeError(f"Using 'read_only' and 'restart' together is not allowed. Restarting training will overwrite the checkpoint file '{cache_ffn}'.")
            raise RuntimeError(f"Checkpoint '{cache_ffn}' does not exist. Cannot load model.")
        verbose.write(lambda : f"Using recovery checkpoint '{cache_ffn}'.")

    # preprocess & copy data into tensors
    # -----------------------------------

    with verbose.write_t(lambda : f"Preparing data set {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        total_tdata      = mdl_config.torch_data(
                                env = env,
                                market = market, # no reload here
                                message_context = None,
                                verbose = Context.quiet  )
        trn_tdata        = { k: v[:num_trn_samples,...] if v.shape[0]>1 else v for k, v in total_tdata.items() }
        vld_tdata        = { k: v[num_trn_samples:,...] if v.shape[0]>1 else v for k, v in total_tdata.items() } if num_vld_samples > 0 else None
        verbose.write(lambda : f"done; this took {tme}.", head=False )

    with verbose.write_t(lambda : f"\rComputing loss... ", end='') as tme:        
        trn_ld0 = model( trn_tdata, training=False )
        vld_ld0 = model( vld_tdata, training=False ) if vld_tdata is not None else None
        str_vld = f" and validation loss {vld_ld0['loss']:.4g}" if not vld_ld0 is None else ""
        verbose.write(lambda : f"\n{'Initial' if start_epoch == 0 else 'Current'} training loss is {trn_ld0['loss']:.4g}{str_vld}. This took {tme}.", head=False )

    def sample():
        if trn_config.batch_size is None or trn_config.batch_size >= num_trn_samples:
            return trn_tdata, None
        else:
            idx = torch.randint(0, num_trn_samples, (trn_config.batch_size,))
            return { k: v[idx,...] if not v is None else None for k, v in trn_tdata.items() }, idx

    # compile
    # -------

    def gradient_step( batch ):
        loss = model( batch, training=True ) 
        model.verify( lambda : torch.isfinite(loss).all(), "Internal error: loss infinite" )
        return loss

    def valuation( all_data ):
        with torch.no_grad():
            return model( all_data, training=False )

    if start_epoch < trn_config.num_epochs and trn_config.compile:
        with verbose.write_t(lambda : f"Compiling {mdl_config.label} version {mdl_config.user_version} for {env.device}: training... ", end='') as tme:
            gradient_step = torch.compile( mode="reduce-overhead",
                            dynamic=False,
                            fullgraph=True,
                            disable=model.debug,
                        )( gradient_step )
            _ = gradient_step(sample()[0])
            del _

            verbose.write("valuation... ", head=False, end='')

            valuation = torch.compile( mode="reduce-overhead",
                            dynamic=False,
                            fullgraph=True,
                            disable=model.debug,
                        )( valuation )
            _ = valuation(trn_tdata)
            if not vld_tdata is None:
                verbose.write("validation... ", head=False, end='')
                _ = valuation(vld_tdata)
            del _
            verbose.write(lambda : f"done; this took {tme}.", head=False ) 

    # visualization & seralization
    # ----------------------------

    disp = Visualizer( market=market, 
                       model_output0=PrettyObject({k:v.detach().cpu().numpy() for k,v in trn_ld0.items() if not v is None}),
                       viz_config=viz_config,
                       mdl_config=mdl_config
                  )
    del trn_ld0, vld_ld0

    def viz_and_store(*, epoch: int, final : bool = False):
        """ call valuation and then visualize and store the model """
        model.eval()
        trn_ld = valuation( trn_tdata )
        vld_ld = valuation( vld_tdata ) if not vld_tdata is None else None
        assert len(tracker.val_epochs) == 0 or tracker.val_epochs[-1] <= epoch+1, f"Expected last validation epoch {tracker.val_epochs[-1]} to be less than or equal to current epoch {epoch+1}"
        if len(tracker.val_epochs)==0 or tracker.val_epochs[-1] < epoch+1:
            tracker.val_losses.append(trn_ld['loss'].cpu().item())
            tracker.val_epochs.append(epoch+1)
        else:
            tracker.val_losses[-1] = trn_ld['loss'].cpu().item()
            tracker.val_epochs[-1] = epoch+1
        if not vld_ld is None:
            if len(tracker.vld_epochs)==0 or tracker.vld_epochs[-1] < epoch+1:
                tracker.vld_losses.append(vld_ld['loss'].cpu().item())
                tracker.vld_epochs.append(epoch+1)
            else:
                tracker.vld_losses[-1] = vld_ld['loss'].cpu().item()
                tracker.vld_epochs[-1] = epoch+1

        # convert to numpy
        def compute( ld ):
            if ld is None:
                return None, None
            model_output = PrettyObject({k:v.detach().cpu().numpy() for k,v in ld.items() if not v is None})
            weights      = [ np.mean( market.weights[i,...][market.weights[i,...]>0] ) for i in range(market.weights.shape[0]) ]
            errors       = [ model_output.errors[i,...][market.weights[i,...]>0] for i in range(model_output.errors.shape[0]) ]
            raw_errors   = [ model_output.raw_errors[i,...][market.weights[i,...]>0] for i in range(model_output.raw_errors.shape[0]) ]

            # per sample summary
            # raw_errors = abs( weight*(fit-mid) ) 
            # errors = raw_errors**2 if raw_errors<1 else 2*raw_errors - 1 (huber loss)
            es = PrettyObject()
            es.error_stats = PrettyObject()
            es.error_stats.weights     = np.array(weights)
            es.error_stats.medians     = np.array([ np.median(err) for err in errors ])
            es.error_stats.means       = np.array([ np.mean(err) for err in errors ])
            es.error_stats.p90         = np.array([ np.quantile(err, 0.9) for err in errors ])
            es.raw_error_stats         = PrettyObject()
            es.raw_error_stats.weights = np.array(weights)
            es.raw_error_stats.medians = np.array([ np.median(err) for err in raw_errors ])
            es.raw_error_stats.means   = np.array([ np.mean(err) for err in raw_errors ])
            es.raw_error_stats.p90     = np.array([ np.quantile(err, 0.9) for err in raw_errors ])
            return model_output, es
        
        trn_model_output, trn_es = compute(trn_ld)
        vld_model_output, vld_es = compute(vld_ld)
        
        tmp_name = path.full_temp_file_name(file_name, ext="tmp", create_directory=True)
        torch.save(dict(version=mdl_config.user_version,
                        mdl_version=model_version,
                        tracker=asdict(tracker),
                        optimizer_state_dict=optimizer.state_dict(),
                        model_state_dict=model.state_dict(),
                        grid_market_def=asdict(market.grid_market_def),
                        mdl_config=asdict(mdl_config),
                        mdl_config_class=mdl_config.__class__.__name__,
                        trn_config=asdict(trn_config),
                        error_stats=dict(trn_es.error_stats),
                        raw_error_stats=dict(trn_es.raw_error_stats),
                        vld_error_stats=dict(vld_es.error_stats) if not vld_es is None else {},
                        vld_raw_error_stats=dict(vld_es.raw_error_stats) if not vld_es is None else {},
                        ), tmp_name)
        os.rename(tmp_name, cache_ffn)

        # tracker, num_total_epochs, *, model_output, error_stats, final : bool = False ):
        disp(tracker=tracker, num_total_epochs=trn_config.num_epochs,
                              model_output=trn_model_output, error_stats=trn_es, 
                              #vld_model_output=vld_model_output, vld_error_stats=vld_es,
                              final=final)
        return (trn_model_output, vld_model_output) if final else None

    # train
    # -----

    epoch = start_epoch
    if start_epoch < trn_config.num_epochs:
        with verbose.write_t(lambda : f"Training {mdl_config} with {trn_config} on {env}:") as tme:
            last_update = time.time()
            try:
                for epoch in range(start_epoch, trn_config.num_epochs):

                    # -------------------------
                    # Training
                    # -------------------------
                    model.train()

                    train_loss = 0.0

                    optimizer.zero_grad(set_to_none=True)
                    batch, idx = sample()
                    loss = gradient_step(batch)

                    loss.backward()

                    # Optional but often useful
                    if trn_config.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=trn_config.clip_grad_norm)

                    lr = trn_config.set_lr_for_epoch(epoch, optimizer)
                    optimizer.step()
                    train_loss = loss.detach().item() 
                    tracker.trn_losses.append(train_loss)
                    tracker.trn_epochs.append(epoch+1)

                    if epoch in [0,trn_config.num_epochs-1] or (tme.interval_test(1.) and ((epoch+1) % 10 == 0)):
                        verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Loss: {train_loss:.4g}. So far this took {tme}.", end='')
                        
                    if epoch in [0,trn_config.num_epochs-1] or ((epoch+1) % trn_config.epoch_interval == 0) or ((time.time() - last_update) > trn_config.time_interval):
                        viz_and_store(epoch=epoch)
                        last_update = time.time()
                        if num_vld_samples == 0:
                            verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Trn loss: {train_loss:.4g}. Val loss: {tracker.val_losses[-1]:.4g}. So far this took {tme}. [Check point saved]", end='')
                        else:
                            verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Trn loss: {train_loss:.4g}, "+\
                                            f"Val loss: {tracker.val_losses[-1]:.4g}, "+\
                                            f"Vld loss: {tracker.vld_losses[-1]:.4g}. So far this took {tme}. [Check point saved]", end='')

            except KeyboardInterrupt as e:
                verbose.write(lambda : f"\r*** Training interrupted at epoch {epoch+1}/{trn_config.num_epochs}. This took {tme}. Saving checkpoint and exiting.")
                viz_and_store(epoch=epoch,final=True)
                raise e
        verbose.write(lambda : f"\rModel trained for {trn_config.num_epochs} epochs. This took {tme}.")

    # finalize & return
    # -----------------

    model.eval()
    trn_model_output,\
    vld_model_output = viz_and_store(epoch=epoch,final=True)
    mo_path          = path(ext="mo", fmt=SubDir.PYTREE_HDF5)
    verbose1         = verbose(1)
    verbose1.write(lambda : f"\rModel checkpoint file is'{cache_ffn}'.")
    with verbose1.write_t(lambda : f"\rWriting model output to '{mo_path.full_file_name(file_name)}'... ", end='') as tme: 
        mo_path.write(file_name, trn_model_output, raise_on_error=True)
        verbose1.write(lambda : f"done; this took {tme}.", head=False )

    str_vld = f", validation loss: {tracker.vld_losses[-1]:.4g}" if not vld_model_output is None else ""
    verbose.write(lambda : f"\rModel {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' trained for {trn_config.num_epochs} epochs. "+\
                           f"Final training loss: {tracker.trn_losses[-1]:.4g}, valuation loss: {tracker.val_losses[-1]:.4g}{str_vld}. This took {tme}.")
    return trn_model_output,  model, trn_tdata

def load_model_market( ffn, env : BaseEnvironment = BaseEnvironment(), *, 
                       pool_config : PoolConfig = PoolConfig.off(),
                       verbose : Context = Context.all ):
    """
    Load model and market from a file ``ffn``.
    """
    if not os.path.exists(ffn):
        raise FileNotFoundError(f"File '{ffn}' does not exist")

    with verbose.write_t(lambda : f"Loading model from '{ffn}' on {env}... ", end='') as tme:   
        state = torch.load(ffn, map_location=env.device, weights_only=False)
        verbose.write(lambda : f"done; this took {tme}.", head=False )

    user_version     = state.pop("version", None)
    mdl_version      = state.pop("mdl_version", None)
    mdl_config_class = state.pop("mdl_config_class", None)
    mdl_config       = state.pop("mdl_config", None)
    grid_market_def  = state.pop("grid_market_def", None)
    verify( not (mdl_version is None or mdl_config is None or mdl_config_class is None or grid_market_def is None), lambda : f"File '{ffn}' does not contain a valid model configuration.")

    grid_market_def  = GridMarketDefinition(**grid_market_def)
    mdl_config       = SanosModelConfig(**mdl_config)

    verify(user_version == mdl_config.user_version, lambda : f"File '{ffn}' user_version {user_version} does not match configured model user_version {mdl_config.user_version}.")

    with verbose.write_t(lambda : f"Loading market... ", end='') as tme:   
        market        = grid_market_def.load( pool_config = pool_config, verbose=Context.quiet )
        verbose.write(lambda : f"done; this took {tme}. Found {len(market.dates)} dates.", head=False )

    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        T, N      = market.grid_market_def.TN
        model     = mdl_config.create(  env,
                                        num_expiries=T,
                                        num_market_strikes=N )
        verify(model.version.unique_id48 == mdl_version, lambda : f"File '{ffn}' model version {mdl_version} does not match model version {model.version.unique_id48}.")
        verbose.write(lambda : f"done; this took {tme}.", head=False )
        if model.debug:
            warnings.warn("** WARNING: model is running in (very slow) debug mode. **")

    return model, market
