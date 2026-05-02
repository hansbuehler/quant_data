"""
Dysanos trainer
"""

from __future__ import annotations
from copyreg import pickle
import math
import warnings

from quantdata.dysanos.simple import locations_simple
from quantdata.torch import get, pop, compute_grad_norm, detect_grad_issues, MetricTacker, BaseEnvironment, BaseModel, BaseSampler
from dataclasses import dataclass, asdict
from cdxcore.err import verify_inp, verify
from cdxcore.util import TrackTime, fmt_filename, fmt_seconds, fmt_list
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, version
from cdxcore.uniquehash import named_unique_filename48_8, unique_hash8
from cdxcore.pretty import PrettyObject
import torch as torch
import numpy as nnp
import copy as copy
import time as time
import pickle as pickle

@version("0.1")
@dataclass(frozen=True)
class TrainConfig:
    """
    Training configuration
    """
    trn_batch_size  : int = 16    #: batch size used for training.
    val_batch_size  : int = 64    #: validation batch size; optimize for GPU utilization
    batch_per_epoch : int = 10    #: How many training batches to use per epoch before valuation is run.

    # clipping
    max_grad_norm   : float|None = None #: If not None, clip the gradient norm to this value to prevent exploding gradients. Set to None to disable gradient clipping.

    # optimizer
    lr_init         : float = 1E-3  #: Initial learning rate
    lr_final        : float = 1E-6  #: Final learning rate after decay
    lr_init_steps   : int = 10      #: Number of steps (epochs) to keep the learning rate at lr_init before starting decay
    lr_decay_steps  : int = 1000    #: Number of steps (epochs) over which to decay the learning rate from lr_init to lr_final

    # etc
    seed            : int = 123123  #: Random seed for reproducibility. This controls the random sampling of training batches, but does not affect the model initialization, which should be controlled by the model itself.
    chkpnt_epochs   : int = 10      #: After how many epochs to write a checkpoint
    chkpnt_minutes  : float = 1.    #: After how many minutes to write a checkpoint
    experiment_group: str = ""      #: Subdirectory for check points
    experiment_root : str|SubDir|None = None #: The root training directory, overwriting the system default ``locations_simple.experiments``.

    def __post_init__(self):
        verify_inp(self.trn_batch_size > 0, lambda : f"'trn_batch_size' must be a positive integer; found {self.trn_batch_size}")
        verify_inp(self.val_batch_size > 0, lambda : f"'val_batch_size' must be a positive integer; found {self.val_batch_size}")
        verify_inp(self.batch_per_epoch > 0, lambda : f"'batch_per_epoch' must be a positive integer; found {self.batch_per_epoch}")
        verify_inp(self.lr_init > 0., lambda : f"'lr_init' must be a positive float; found {self.lr_init}")
        verify_inp(self.lr_final > 0., lambda : f"'lr_final' must be a positive float; found {self.lr_final}")
        verify_inp(self.lr_init_steps >= 0, lambda : f"'lr_init_steps' must be a non-negative integer; found {self.lr_init_steps}")
        verify_inp(self.lr_decay_steps > 0, lambda : f"'lr_decay_steps' must be a positive integer; found {self.lr_decay_steps}")
        verify_inp(self.chkpnt_epochs > 0, lambda : f"'chkpnt_epochs' must be a positive integer; found {self.chkpnt_epochs}")
        verify_inp(self.chkpnt_minutes > 0., lambda : f"'chkpnt_minutes' must be a positive float; found {self.chkpnt_minutes}")

    def learning_rate( self, epoch ) -> callable:
        """ Return function to compute learning rate based on the initial epoch """
        if epoch <= self.lr_init_steps:
            return self.lr_init
        elif epoch >= self.lr_init_steps + self.lr_decay_steps:
            return self.lr_final
        else:
            decay_factor = (self.lr_final / self.lr_init) ** ((epoch - self.lr_init_steps) / self.lr_decay_steps)
            assert 0. < decay_factor <= 1., f"Internal error: decay_factor should be in (0,1], but got {decay_factor} for epoch {epoch}"
            return decay_factor*self.lr_init

    def unique_id(self) -> str:
        return unique_hash8(self)

    def get_experiment_directory( self, unique_data_id : str, model_label : str|None = None, *, ext : str|None = None, fmt : SubDir.Format|str|None = None ) -> SubDir:
        """
        Get the experiment directory for this training configuration, model label and unique data id.
        
        * The unique data id should uniquely identify the data that is being trained on, and can be obtained from the sampler or its config.
        * The model label should be a human-readable label for the model, which can be generated from the model class name and the training configuration.
          If omitted, this will return the root of the experiment group.
        * The ``fmt`` and ``ext`` arguments are passed to :class:`cdxcore.subdir.SubDir`.
        """
        experiment = SubDir(locations_simple.experiments if self.experiment_root is None else self.experiment_root,
                                  ext=ext, 
                                  fmt=fmt) 
        experiment = experiment(unique_data_id)
        experiment = experiment(self.experiment_group) if self.experiment_group != "" else experiment
        experiment = experiment(named_unique_filename48_8(model_label)) if not model_label is None else experiment
        return experiment

@version("0.1", dependencies=[TrainConfig])
class Trainer(object):
    """ Trainer for dysanos simple """

    def __init__(self, 
                 label   : str|None,
                 model   : BaseModel,
                 sampler : BaseSampler,
                 config  : TrainConfig,
                 restart : bool = False,
                 verbose : Context = Context.all ):
        """
        Main training loop
        """
        label = None if label == "" else label

        self._info_label = model.info_qualified_label(label) # this is for printing user information
        self._label      = label if label is not None else model.default_label
        self._model      = model
        self._sampler    = sampler
        self._config     = config
        self._verbose    = verbose

        # the path is unique per experiment
        # if a label is provided, it is essentially
        # keyed by:
        # * Market data, via ``sampler.unique_data_id``;
        # * The ``experiment_group`` in the training configuration;
        # * The user's ``label`` which defaults to the 
        #   models' qualified name (e.g. keyed by its configuration)
        self._path       = config.get_experiment_directory( sampler.unique_data_id, self._label, ext="pt", fmt=SubDir.PICKLE )

        self._rndg       = torch.Generator(device="cpu")
        self._visualizer = model.visualizer(sampler) # might be None
        self._rndg.manual_seed(config.seed)
        self._optimizer  = torch.optim.Adam(model.parameters(), lr=config.lr_init)

        self._track_val_trn = MetricTacker()
        self._track_val_vld = MetricTacker()
        self._current_epoch = 0
        self._best_epoch    = None
        self._best_val_loss = None
        self._tracker       = TrackTime()

        if verbose.shall_report():
            s = f"Trainer for '{self.info_label}' on data {sampler.unique_data_id}"
            verbose.write(s+'\n'+("="*len(s)))
            del s
        if model.debug:
            warnings.warn("** WARNING: model is running in (very slow) debug mode. **")

        deleted_best = False
        if not self._path is None and restart and self._path.exists("last"):
            deleted_best = self._path.exists("best")
            self._path.delete("last")
            self._path.delete("best") if deleted_best else None
            verbose.write(lambda : f"'{self.info_label}': Deleted existing recovery checkpoint '{self._path.full_file_name('last')}'{' and the best-state checkpoint' if deleted_best else ''}.")

        if not self._path is None and self._path.exists("last"):
            def restore(verbose):
                ffn = self._path.full_file_name('last')
                with verbose.write_t(f"Restoring checkpoint from '{ffn}'... ", end='') as tme:
                    state         = torch.load(ffn, map_location=model.device, weights_only=False)
                    lbl           = state.pop("label", None)
                    assert not lbl is None, f"Internal error: checkpoint '{ffn}' is missing the 'label' field."
                    if lbl != self._label:
                        raise RuntimeError(f"Internal error '{ffn}': checkpoint label '{lbl}' does not match model label '{self._label}' ?")

                    version       = state.pop("version", "<unknown>") # for backward compatibility with checkpoints that do not contain the version field
                    if version != model.version:
                        verbose.write(f"checkpoint version '{version}' does not match model version '{model.version}' - restarting experiment.", head=False)
                        return

                    self._current_epoch = int(state.pop("epoch"))
                    self._best_epoch    = state.pop("best_epoch", None) if not deleted_best else None
                    self._best_val_loss = state.pop("best_val_loss", None) if not deleted_best else None

                    self._track_val_trn = MetricTacker.load_state_dict(state.pop("track_val_trn"))
                    self._track_val_vld = MetricTacker.load_state_dict(state.pop("track_val_vld"))

                    self._model.load_state_dict(state.pop("model_state_dict"))
                    self._optimizer.load_state_dict(state.pop("optimizer_state_dict"))
                    rndg_state = state.pop("rndg_state_dict")
                    if not isinstance(rndg_state, torch.Tensor):
                        rndg_state = torch.as_tensor(rndg_state, dtype=torch.uint8)
                    else:
                        rndg_state = rndg_state.detach().cpu().to(dtype=torch.uint8)
                    self._rndg.set_state(rndg_state)
                    verbose.write(f"done; this took {tme}.", head=False)
            restore(verbose(1))
        verbose.report(1,lambda : f"Model is ready. Current epoch to train is {self._current_epoch}.")
        verbose.report(1,lambda : f"Experiment path is {self._path.path}.") 

    @property
    def info_label(self) -> str:
        """ label for user information """
        return self._info_label
    
    @property
    def label(self) -> str:
        """Label for the model, as provided by the user or generated from the model class name."""
        return self._label

    @property
    def device(self):
        return self._model.device
    
    def get_state(self) -> dict:
        """ Return current state as dictionary """
        return {
            "label": self._label, # user-specififed label
            "epoch": int(self._current_epoch),
            "version": self._model.version,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "rndg_state_dict": self._rndg.get_state(),
            "track_val_trn": self._track_val_trn.state_dict(),
            "track_val_vld": self._track_val_vld.state_dict(),
            "best_epoch": self._best_epoch,
            "best_val_loss": self._best_val_loss,
            "sampler_config": pickle.dumps(self._sampler.config),
            "model_config": pickle.dumps(self._model.config),
            "model_env": pickle.dumps(self._model.environment)
        }
    
    @staticmethod   
    def instantiate_model_and_sampler_from_cache( path: SubDir|str, device : torch.device|str|None = "cpu", *, best : bool = False, verbose : Context = Context.quiet ) -> dict:
        """ load state dictionary """
        try:
            path   = SubDir(path, ext="pt")
            ffn    = path.full_file_name("last" if not best else "best")
            with verbose.write_t(lambda : f"Loading {'best' if best else 'last'} checkpoint from '{ffn}'... ", end='') as tme:
                state   = torch.load(ffn, map_location=device, weights_only=False)
                verbose.write(lambda : f"done; this took {tme}.", head=False)

            # instantiate sampler and model
            r = PrettyObject()
            r.label          = state.pop("label")
            assert not r.label is None, f"Internal error: checkpoint '{ffn}' is missing the 'label' field."
            r.version        = state.pop("version")
            r.epoch          = int(state.pop("epoch"))
            r.best_epoch     = state.pop("best_epoch", None)
            r.best_val_loss  = state.pop("best_val_loss", None)
            r.track_val_trn  = MetricTacker.load_state_dict(state.pop("track_val_trn"))
            r.track_val_vld  = MetricTacker.load_state_dict(state.pop("track_val_vld"))
            r.sampler_config = pickle.loads(state.pop("sampler_config"))
            r.model_config   = pickle.loads(state.pop("model_config"))
            r.env            = pickle.loads(state.pop("model_env"))
            if not device is None:
                r.env.device = device
            r.sampler        = r.sampler_config.instantiate_from_cache(verbose=verbose(1))
            r.model          = r.model_config.instantiate_from_cache(env=r.env, sampler=r.sampler, verbose=verbose(1))
            r.model.load_state_dict(state.pop("model_state_dict"))
            r.optimizer.load_state_dict(state.pop("optimizer_state_dict"))
        except Exception as e:
            raise type(e)(f"Failed to instantiate model and sampler from checkpoint '{ffn}': {str(e)}") from e
        verbose.write(lambda : f"Model '{r.model.info_label}' instantiated from checkpoint '{ffn}' for epoch {r.epoch} with best validation loss {r.best_val_loss} at epoch {r.best_epoch} on device {r.env.device}.")
        return r

    def __call__( self, epochs : int, *, total : bool = True ) -> BaseModel:
        """ Run model """
        verbose = self._verbose
        model   = self._model
        tracker = self._tracker
        config  = self._config
        sampler = self._sampler

        epochs = epochs if total else  ( self._current_epoch + epochs )
        if self._current_epoch >= epochs:
            verbose.write(lambda : f"Model '{self.info_label}' was already trained for {self._current_epoch} epochs.")
            if self._visualizer is not None:
                self._visualizer.update(model,epoch=self._current_epoch, trn_tracker=self._track_val_trn, vld_tracker=self._track_val_vld) 
            return model

        verbose.write(lambda : f"Launching '{self.info_label}' training for {epochs} epochs over {self._sampler.num_trn_samples} training and {self._sampler.num_vld_samples} validation samples, using a mini bastch size of "+\
                               f"{config.trn_batch_size} for {config.batch_per_epoch} batches per epoch, and a batch size of {config.val_batch_size} for valuation. "+\
                               f"The model has {sum(p.numel() for p in model.parameters())} parameters." +\
                               (f" Resuming from epoch {self._current_epoch}." if self._current_epoch>0 else ""))

        # training
        # --------

        last_checkpoint_time = None

        def set_lr(lr: float) -> None:
            for group in self._optimizer.param_groups:
                group["lr"] = lr

        for epoch in range(self._current_epoch+1, epochs+1):
            assert all(torch.isfinite(p).all().item() for p in model.parameters()), "Model parameters contain non-finite values before training step"
            grad_norm = 0.
            lr = config.learning_rate(epoch)
            set_lr(lr)

            with tracker("training"):
                def train():
                    tme    = verbose.timer()
                    loss   = 0.
                    weight = 0.
                    model.train()
                    for imini in range(config.batch_per_epoch):
                        verbose.write(lambda : f"\rTraining epoch {epoch}/{epochs} mini batch {imini+1}/{config.batch_per_epoch} with learning rate {lr:.2e}: so far this took {tme}...", end='')

                        batch = sampler( self._rndg, trn=True, batch_size=config.trn_batch_size, device=self.device, return_batch_indices=False )
                        assert batch['is_valid'].any().item(), "Sampled batch contains no valid samples. Check the sampler and the data."

                        self._optimizer.zero_grad(set_to_none=True)

                        assert all(torch.isfinite(p).all().item() for p in model.parameters()), f"Model parameters contain non-finite values before model call {epoch} {imini}"
                        ls, wg = model( batch, training=True )
                        mini_loss = ls / wg
                        assert torch.isfinite(mini_loss).all(), f"Internal error: mini_loss infinite"

                        mini_loss.backward()

                        detect_grad_issues(model)

                        if config.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        grad_norm = compute_grad_norm(model)

                        self._optimizer.step()
                        assert all(torch.isfinite(p).all().item() for p in model.parameters()), f"Model parameters contain non-finite values after optimizer step {epoch} {imini}"

                        loss    += ls.detach().item()
                        weight  += wg.detach().item()
                    return loss / weight, grad_norm

                trn_loss, grad_norm = train()
                
            if verbose.shall_report():
                str_tme_trn = f"{tracker['training'].fmt_average_lap_seconds()}"
                verbose.write(f"\rTraining epoch {epoch}/{epochs} error avg trn {trn_loss:.4g} [{str_tme_trn}] learning rate {lr:.2e} grad norm {grad_norm:.4g}... running valuation... ", end='')

            # valuation
            # ---------

            trn_val_loss = None
            vld_val_loss = None

            with torch.no_grad():
                def value( trn ):
                    loss    = 0.
                    loss_sq = 0.
                    weight  = 0.
                    metrics = {}

                    for batch in sampler.all( trn=trn, batch_size=config.val_batch_size, device=self.device ):
                        ls, lsq, wg, lm = model( batch, training=False, return_what=model.ReturnWhat.VALUATION )
                        verify( not 'loss' in lm, lambda : f"Setup error '{self.info_label}': model returned 'loss' as a metric, but 'loss' should be returned as a separate value from the forward method, not as a metric." )
                        ls      = ls.detach().item()
                        lsq     = lsq.detach().item()
                        wg      = wg.detach().item()
                        assert nnp.isfinite(ls+lsq+wg) and lsq >=0. and wg > 0, f"Internal error: {ls} {lsq} {wg}"
                        loss    += ls
                        loss_sq += lsq
                        weight  += wg

                        # each metric contains a nominator and denominator, to be aggregated accross all samples
                        for metric, mvalues in lm.items():
                            verify( len(mvalues) in [2,3], lambda : f"Setup error '{self.info_label}': metric '{metric}' has returned {len(mvalues)}, but two or three are required: E[w x], E[w x^2] and E[w]" )
                            mvalues = [ m.detach().item() for m in mvalues ]
                            assert all(nnp.isfinite(x) for x in mvalues), f"Internal error: metric '{metric}' has non-finite or negative values: {mvalues}"
                            assert mvalues[-1] > 0, f"Internal error: metric '{metric}' must have posititive weight, found {mvalues[-1]}"
                            assert len(mvalues) == 2 or mvalues[1] >= 0, f"Internal error: metric '{metric}' must have non-negative 'square' value; found {mvalues[1]}"

                            xvalues = metrics.get( metric, None )
                            xvalues = [ x + m for x,m in zip(xvalues, mvalues) ] if not xvalues is None else mvalues
                            assert xvalues[-1] > 0., f"Internal error: metric '{metric}' has non-positive total weight {xvalues[-1]} after aggregation"
                            metrics[metric] = xvalues
                        del metric, mvalues, xvalues
    
                    # take averages
                    loss = loss / weight
                    loss_std = math.sqrt( max(0., loss_sq / weight - loss*loss) )
                    del weight, loss_sq

                    agg_metrics = dict(loss=(loss, loss_std))
                    for metric, mvalues in metrics.items():
                        if len(mvalues) == 3:
                            Ex   = mvalues[0]
                            Exsq = mvalues[1]
                            w    = mvalues[2]
                            assert w > 0., f"Internal error: metric '{metric}' has non-positive total weight {w} after aggregation"
                            mean = Ex / w
                            std  = math.sqrt( max(0., Exsq / w - mean*mean) )
                            agg_metrics[metric] = ( mean, std )
                        else:
                            assert len(mvalues) == 2
                            assert mvalues[1] > 0., f"Internal error: metric '{metric}' has non-positive total weight {mvalues[1]} after aggregation"
                            agg_metrics[metric] = ( mvalues[0] / mvalues[1], None )

                    return loss, agg_metrics
                
                with tracker("val_trn"):
                    trn_val_loss, trn_metrics = value(trn=True)
                    self._track_val_trn.record( epoch, trn_metrics )
                    self._track_val_trn.record_metric( epoch, "lr", lr )
                with tracker("val_vld"):
                    vld_val_loss, vld_metrics = value(trn=False)    
                    self._track_val_vld.record( epoch, vld_metrics )
                    self._track_val_vld.record_metric( epoch, "lr", lr )

            assert not trn_val_loss is None and not vld_val_loss is None, "Internal error: valuation failed to produce a validation loss"

            # print
            # -----

            if verbose.shall_report():
                str_tme_val_trn = f"{tracker['val_trn'].fmt_average_lap_seconds()}"
                str_tme_val_vld = f"{tracker['val_vld'].fmt_average_lap_seconds()}"   
                str_estimated   = fmt_seconds( tracker.current_lap_seconds * (epochs - epoch) )

                str_metrics = f"LR: {lr:.2e}, GN {grad_norm:.4g}, "
                for k in trn_metrics:
                    if k!="loss":
                        trn_v = trn_metrics[k][0]
                        vld_v = vld_metrics[k][0]
                        str_metrics += f"{k}: {trn_v:.4g}/{vld_v:.4g}, "
                str_metrics = str_metrics[:-2] + "."
                verbose.write(f"\rTraining '{self.label}' epoch {epoch}/{epochs} error avg trn {trn_loss:.4g}; val {trn_val_loss:.4g}/{vld_val_loss:.4g} "+\
                              f"[{str_tme_trn} {str_tme_val_trn} {str_tme_val_vld}]. {str_metrics} ETA: {str_estimated}.")

            # checkpointing
            # -------------

            self._current_epoch = epoch
            if not self._path is None and\
                ( last_checkpoint_time is None or\
                  epoch % config.chkpnt_epochs == 0 or\
                  epoch >= epochs or\
                  last_checkpoint_time is None or\
                  config.chkpnt_minutes*60. <= (time.time() - last_checkpoint_time)):
                verbose1 = verbose(1)
                state    = None
                if not vld_val_loss is None and (self._best_val_loss is None or vld_val_loss < self._best_val_loss):
                    ffn = self._path.full_file_name("best")
                    str_prev = f" vs. {self._best_val_loss:.4g} at epoch {self._best_epoch}" if self._best_val_loss is not None else ""
                    with verbose1.write_t(f"Epoch {epoch}: Saving new best validation loss {vld_val_loss:.4g}{str_prev} to '{ffn}'... ", end='') as tme:
                        self._best_epoch = epoch
                        self._best_val_loss = vld_val_loss
                        self._path.create_directory()
                        state = self.get_state()
                        torch.save(state, ffn)
                        verbose1.write(f"done; this took {tme}.", head=False)

                ffn = self._path.full_file_name("last")
                with verbose1.write_t(f"Saving checkpoint to {ffn}... ", end='') as tme:
                    self._path.create_directory()
                    torch.save(self.get_state() if state is None else state, ffn)
                    verbose1.write(f"done; this took {tme}.", head=False)

                last_checkpoint_time = time.time()

            # visualize
            # ---------

            if self._visualizer is not None:
                self._visualizer.update(model,epoch=epoch, trn_tracker=self._track_val_trn, vld_tracker=self._track_val_vld) 

        return model
