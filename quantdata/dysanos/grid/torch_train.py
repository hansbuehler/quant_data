from dataclasses import dataclass, field, asdict
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir
from cdxcore.err import verify_inp
from cdxcore.version import version
from cdxcore.uniquehash import unique_label48_8
import warnings
from quantdata.dysanos.grid.torch_model import SanosModelConfig, BaseEnvironment
from quantdata.dysanos.grid.torch_gfx import VizConfig, Visualizer
import torch as torch
import time as time
import numpy as np
import os
import ipynbname

_cache_root = SubDir("~/.dysanos_grid/")

@dataclass(frozen=True)
class TrainConfig:
    num_epochs          : int = 5000      # total epochs
    batch_size          : int|None = None # batch size, or None for all samples
    lr                  : float = 1E-3    # initial learning rate
    final_lr            : float = 1E-5    # final learning rate
    weight_decay        : float = 1e-4    # weight decay for regularization
    compile             : bool = False    # speed up training dramatically

    epoch_interval : int = 200   # after how many epochs to update visualization and store checkpoint
    time_interval : float = 2*60   # after how many seconds to update visualization and store checkpoint

    def __post_init__(self):
        verify_inp( self.num_epochs > 0, f"Expected 'num_epochs' to be positive, but found {self.num_epochs}" )
        verify_inp( self.final_lr > 0., f"Expected 'final_lr' to be positive, but found {self.final_lr}" )
        verify_inp( self.lr >= self.final_lr, f"Expected 'lr' to be greater than or equal to 'final_lr' {self.final_lr}, but found {self.lr}" )
        verify_inp( self.weight_decay >= 0., f"Expected 'weight_decay' to be non-negative, but found {self.weight_decay}" )
        verify_inp( self.epoch_interval > 0, f"Expected 'epoch_interval' to be positive, but found {self.epoch_interval}" )
        verify_inp( self.time_interval > 0., f"Expected 'time_interval' to be positive, but found {self.time_interval}" )

    def __str__(self):
        return f"epochs={self.num_epochs},lr={self.lr:.4g}->{self.final_lr:.4g},wdec={self.weight_decay:.4g},bsize={'all' if self.batch_size is None else self.batch_size}"

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

@dataclass
class MetricTracker:
    trn_epochs : list[int] = field(default_factory=list)
    trn_losses : list[float] = field(default_factory=list)
    val_epochs : list[int] = field(default_factory=list)
    val_losses : list[float] = field(default_factory=list)

def train( market        : PrettyObject,*,
           mdl_config    : SanosModelConfig, 
           trn_config    : TrainConfig = TrainConfig(),
           viz_config    : VizConfig = VizConfig(),
           restart       : bool = False,
           env           : BaseEnvironment = BaseEnvironment(),
           verbose       : Context = Context.all ) -> dict:
    """
    Train model
    """

    verbose.write(lambda : f"Training {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' for {trn_config.num_epochs} epochs on {market.grid_market_def.ticker} for date range {market.grid_market_def.date_range}")
    verbose.write(lambda : f"Using {'all samples' if trn_config.batch_size is None else f'batches of {trn_config.batch_size} samples'} with learning rate {trn_config.lr:.4g} and weight decay {trn_config.weight_decay:.4g}.")
    verbose.report(1,lambda : f"Model config: {mdl_config}")
    verbose.report(1,lambda : f"Grid config: {market.grid_market_def.grid_config}")
    verbose.report(1,lambda : f"Mk Arb Free pre-processor: {market.grid_market_def.mk_arb_free}")
    verbose.report(1,lambda : f"Options filter: {market.grid_market_def.options_filter}")
    verbose.report(1,lambda : f"Raw day filter: {market.grid_market_def.raw_day_filter}")
    verbose.report(1,lambda : f"Expiry filter: {market.grid_market_def.expiry_filter}")

    # preprocess & copy data into tensors
    # -----------------------------------

    with verbose.write_t(lambda : f"Preparing data set {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        tdata      = mdl_config.torch_data(
                                env = env,
                                market = market, # no reload here
                                message_context = None,
                                verbose = Context.quiet  )
        verbose.write(lambda : f"done; this took {tme}.", head=False )

    # create model
    # ------------

    def sample():
        if trn_config.batch_size is None or trn_config.batch_size >= S:
            return tdata, None
        else:
            idx = torch.randint(0, S, (trn_config.batch_size,))
            return { k: v[idx,...] if not v is None else None for k, v in tdata.items() }, idx
        

    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} on {env}... ", end='') as tme:
        S, T, N    = tdata['mids'].shape
        model = mdl_config.create(  env,
                                    S=S,
                                    T=T,
                                    N=N )
        ld0 = model( tdata, training=False )
        verbose.write(lambda : f"done; initial loss is {ld0['loss']:.4g}. This took {tme}.", head=False )
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
        if user_version is None:
            warnings.warn(f"Checkpoint '{cache_ffn}' does not contain a 'version' field. This is likely an old checkpoint. Restarting training.")
        elif user_version != mdl_config.user_version:
            warnings.warn(f"Checkpoint '{cache_ffn}' file user_version {user_version} does not match model user_version {mdl_config.user_version}. Restarting training.")
        elif not mdl_version is None and model_version != mdl_version:
            warnings.warn(f"Checkpoint '{cache_ffn}' file model version {mdl_version} does not match model version {model_version}. Restarting training.")
        else:
            optimizer.load_state_dict(state.pop("optimizer_state_dict"))
            model.load_state_dict(state.pop("model_state_dict"))
            tracker = MetricTracker(**state.pop("tracker"))
            start_epoch = tracker.trn_epochs[-1] + 1

    else:
        verbose.write(lambda : f"Using recovery checkpoint '{cache_ffn}'.")

    # compile
    # -------

    def gradient_step( batch, idx ):
        loss = model( batch, idx, training=True ) 
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
            _ = gradient_step(*sample())
            del _

            verbose.write("valuation... ", head=False, end='')

            valuation = torch.compile( mode="reduce-overhead",
                            dynamic=False,
                            fullgraph=True,
                            disable=model.debug,
                        )( valuation )
            _ = valuation(tdata)
            del _
            verbose.write(lambda : f"done; this took {tme}.", head=False ) 

    # visualization & seralization
    # ----------------------------

    disp = Visualizer( market=market, 
                       model_output0=PrettyObject({k:v.detach().cpu().numpy() for k,v in ld0.items() if not v is None}),
                       viz_config=viz_config,
                       mdl_config=mdl_config
                  )
    del ld0

    def viz_and_store(*, epoch: int, final : bool = False):
        """ call valuation and then visualize and store the model """
        model.eval()
        ld = valuation( tdata )
        assert len(tracker.val_epochs) == 0 or tracker.val_epochs[-1] <= epoch+1, f"Expected last validation epoch {tracker.val_epochs[-1]} to be less than or equal to current epoch {epoch+1}"
        if len(tracker.val_epochs)==0 or tracker.val_epochs[-1] < epoch+1:
            tracker.val_losses.append(ld['loss'].cpu().item())
            tracker.val_epochs.append(epoch+1)
        else:
            tracker.val_losses[-1] = ld['loss'].cpu().item()
            tracker.val_epochs[-1] = epoch+1

        # convert to numpy
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

        path.create_directory()
        torch.save(dict(version=mdl_config.user_version,
                        mdl_version=model_version,
                        tracker=asdict(tracker),
                        optimizer_state_dict=optimizer.state_dict(),
                        model_state_dict=model.state_dict(),
                        grid_market_def=asdict(market.grid_market_def),
                        mdl_config=asdict(mdl_config),
                        mdl_config_class=mdl_config.__class__.__name__,
                        trn_config=asdict(trn_config),
                        error_stats=dict(es.error_stats),
                        raw_error_stats=dict(es.raw_error_stats),
                        ), cache_ffn + ".tmp")
        os.rename(cache_ffn + ".tmp", cache_ffn)
        
        disp(tracker=tracker, num_total_epochs=trn_config.num_epochs, model_output=model_output, error_stats=es, final=final)
        return model_output if final else None

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
                    n_train = 0

                    optimizer.zero_grad(set_to_none=True)
                    batch, idx = sample()
                    loss = gradient_step(batch,idx)

                    loss.backward()

                    # Optional but often useful
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                        verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Loss: {train_loss:.4g}. Valuation loss: {tracker.val_losses[-1]:.4g}. So far this took {tme}. [Check point saved]", end='')

            except KeyboardInterrupt as e:
                verbose.write(lambda : f"\r*** Training interrupted at epoch {epoch+1}/{trn_config.num_epochs}. This took {tme}. Saving checkpoint and exiting.")
                viz_and_store(epoch=epoch,final=True)
                raise e
        verbose.write(lambda : f"\rModel trained for {trn_config.num_epochs} epochs. This took {tme}.")

    # finalize & return
    # -----------------

    r = viz_and_store(epoch=epoch,final=True)
    verbose.write(lambda : f"\rModel {mdl_config.label} version {mdl_config.user_version} group '{mdl_config.group}' trained for {trn_config.num_epochs} epochs. Final training loss: {tracker.trn_losses[-1]:.4g}, valuation loss: {tracker.val_losses[-1]:.4g}. This took {tme}.\nCheckpoint file is {cache_ffn}.")
    model.eval()
    return r,  model, tdata
