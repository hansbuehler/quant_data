from dataclasses import dataclass, field, asdict
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir
import warnings
from quantdata.dysanos.grid.torch_model import SanosModelConfig, BaseEnvironment
from quantdata.dysanos.grid.torch_gfx import VizConfig, Visualizer
import torch as torch
import time as time
import numpy as np

_cache_root = SubDir("~/.dysanos_grid/")

@dataclass(frozen=True)
class TrainConfig:
    num_epochs : int = 5000
    lr : float = 1E-3
    weight_decay : float = 1e-4
    batch_size : int|None = None

    epoch_interval : int = 200   # after how many epochs to update visualization and store checkpoint
    time_interval : float = 2*60   # after how many seconds to update visualization and store checkpoint

    def __str__(self):
        return f"epochs={self.num_epochs},lr={self.lr:.4g},wdec={self.weight_decay:.4g},bsize={'all' if self.batch_size is None else self.batch_size}"

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

    verbose.write(lambda : f"Training {mdl_config.label} version {mdl_config.user_version} for {trn_config.num_epochs} epochs on {market.grid_market_def.ticker} for date range {market.grid_market_def.date_range}")
    verbose.write(lambda : f"Using {'all samples' if trn_config.batch_size is None else f'batches of {trn_config.batch_size} samples'} with learning rate {trn_config.lr:.4g} and weight decay {trn_config.weight_decay:.4g}.")
    verbose.report(1,lambda : f"Model config: {mdl_config}")
    verbose.report(1,lambda : f"Grid config: {market.grid_market_def.grid_config}")
    verbose.report(1,lambda : f"Mk Arb Free pre-processor: {market.grid_market_def.mk_arb_free}")
    verbose.report(1,lambda : f"Options filter: {market.grid_market_def.options_filter}")
    verbose.report(1,lambda : f"Raw day filter: {market.grid_market_def.raw_day_filter}")
    verbose.report(1,lambda : f"Expiry filter: {market.grid_market_def.expiry_filter}")

    # preprocess & copy data into tensors
    # -----------------------------------

    with verbose.write_t(lambda : f"Preparing data set {mdl_config.label} version {mdl_config.user_version} for {mdl_config} traing {trn_config} on {env}... ", end='') as tme:
        tdata      = mdl_config.torch_data(
                                env = env,
                                market = market, # no reload here
                                message_context = None,
                                verbose = Context.quiet  )
        verbose.write(lambda : f"done; this took {tme}.", head=False )

    # create model
    # ------------

    with verbose.write_t(lambda : f"Instantiating {mdl_config.label} version {mdl_config.user_version} for {mdl_config} traing {trn_config} on {env}... ", end='') as tme:
        S, T, N    = tdata['mids'].shape
        model = mdl_config.create(  env,
                                    S=S,
                                    T=T,
                                    N=N )
        ld = model( tdata, training=False )
        verbose.write(lambda : f"done; initial loss is {ld['loss']:.4g}. This took {tme}.", head=False )
        if model.debug:
            warnings.warn("** WARNING: model is running in (very slow) debug mode. **")
        del ld

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trn_config.lr,
        weight_decay=trn_config.weight_decay
    )

    # load
    # ----

    path        = _cache_root(model.__class__.__name__)(mdl_config.label, ext="pt")
    cache_ffn   = path.full_file_name('last')
    start_epoch = 0
    tracker     = MetricTracker()

    if not restart and path.exists("last"):
        verbose.write(lambda : f"Found existing recovery checkpoint '{cache_ffn}'. Restoring model state from this checkpoint.")

        state = torch.load(cache_ffn, map_location=env.device, weights_only=False)

        user_version = state.pop("version", None)
        if user_version is None:
            warnings.warn(f"Checkpoint '{cache_ffn}' does not contain a 'version' field. This is likely an old checkpoint. Restarting training.")
        elif user_version != mdl_config.user_version:
            warnings.warn(f"Checkpoint '{cache_ffn}' file user_version {user_version} does not match model user_version {mdl_config.user_version}. Restarting training.")
        else:
            optimizer.load_state_dict(state.pop("optimizer_state_dict"))
            model.load_state_dict(state.pop("model_state_dict"))
            tracker = MetricTracker(**state.pop("tracker"))
            start_epoch = tracker.trn_epochs[-1] + 1

    else:
        verbose.write(lambda : f"Using recovery checkpoint '{cache_ffn}'.")

    # train
    # -----

    disp = Visualizer( market=market, 
                       viz_config=viz_config,
                       mdl_config=mdl_config
                  )
    
    with verbose.write_t(lambda : f"Training {mdl_config} with {trn_config} on {env}:") as tme:

        def sample():
            if trn_config.batch_size is None or trn_config.batch_size >= S:
                return tdata, None
            else:
                idx = torch.randint(0, S, (trn_config.batch_size,))
                return { k: v[idx,...] if not v is None else None for k, v in tdata.items() }, idx
            
        def viz_and_store(*, epoch: int, final : bool = False):
            ld = model( tdata, training=False )
            tracker.val_losses.append(ld['loss'].cpu().item())
            tracker.val_epochs.append(epoch+1)

            r = PrettyObject()
            r.xdensities = ld['xdensities'].cpu().numpy()
            r.calls = ld['calls'].cpu().numpy()
            r.dlvsqs = ld['dlvsqs'].cpu().numpy()
            r.state   = ld['state'].cpu().numpy()
            r.errors  = ld['errors'].cpu().numpy()
            r.weights = ld['weights'].cpu().numpy()
 
            # convert 
            weights = [ np.mean( r.weights[i,...][r.weights[i,...]>0] ) for i in range(r.weights.shape[0]) ]
            errors  = [ r.errors[i,...][r.weights[i,...]>0] for i in range(r.errors.shape[0]) ]
            medians = [ np.median(err) for err in errors ]
            means   = [ np.mean(err) for err in errors ]
            p90     = [ np.quantile(err, 0.9) for err in errors ]

            # per sample summary
            r.error_stats = PrettyObject()
            r.error_stats.weights = np.array(weights)
            r.error_stats.medians = np.array(medians)
            r.error_stats.means   = np.array(means)
            r.error_stats.p90     = np.array(p90)

            path.create_directory()
            torch.save(dict(version=mdl_config.user_version,
                            tracker=asdict(tracker),
                            optimizer_state_dict=optimizer.state_dict(),
                            model_state_dict=model.state_dict(),
                            grid_market_def=asdict(market.grid_market_def),
                            mdl_config=asdict(mdl_config),
                            mdl_config_class=mdl_config.__class__.__name__,
                            trn_config=asdict(trn_config),
                            error_stats=dict(r.error_stats),
                            ), cache_ffn)
            
            disp(tracker=tracker, num_total_epochs=trn_config.num_epochs, data=r, final=final)
            return r if final else None

        last_update = time.time()
        epoch = start_epoch
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
                loss = model(batch, idx, training=True)

                loss.backward()

                # Optional but often useful
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss = loss.detach().item() 
                tracker.trn_losses.append(train_loss)
                tracker.trn_epochs.append(epoch+1)

                if epoch in [0,trn_config.num_epochs-1] or (epoch+1) % 10 == 0:
                    verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Loss: {train_loss:.4g}. So far this took {tme}.", end='')
                    
                if epoch in [0,trn_config.num_epochs-1] or (epoch+1) % trn_config.epoch_interval == 0 or (time.time() - last_update) > trn_config.time_interval:
                    viz_and_store(epoch=epoch)
                    last_update = time.time()
                    verbose.report(1, lambda : f"\rEpoch {epoch+1}/{trn_config.num_epochs}, Loss: {train_loss:.4g}. Valuation loss: {tracker.val_losses[-1]:.4g}. So far this took {tme}. [Check point saved]", end='')

        except KeyboardInterrupt as e:
            verbose.write(lambda : f"\r*** Training interrupted at epoch {epoch+1}/{trn_config.num_epochs}. This took {tme}. Saving checkpoint and exiting.")
            viz_and_store(epoch=epoch,final=True)
            raise e

        r = viz_and_store(epoch=epoch,final=True)
        verbose.write(lambda : f"\rModel trained for {trn_config.num_epochs} epochs. Final training loss: {tracker.trn_losses[-1]:.4g}, valuation loss: {tracker.val_losses[-1]:.4g}. This took {tme}.")

    return r

