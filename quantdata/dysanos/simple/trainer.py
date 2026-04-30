"""
Dysanos trainer
"""

from __future__ import annotations

from quantdata.torch import get, pop, compute_grad_norm, detect_grad_issues, MetricTacker, BaseEnvironment, BaseModel
from quantdata.torch.block_sampler import BlockSampler
from dataclasses import dataclass
from cdxcore.err import verify_inp, verify
from cdxcore.util import TrackTime, fmt_seconds, fmt_list
from cdxcore.verbose import Context
from cdxcore.subdir import SubDir, version
from cdxcore.uniquehash import named_unique_filename48_8, unique_hash8
import torch as torch
import numpy as nnp
import copy as copy
import time as time

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

    def learning_rate( self, init_epoch ) -> callable:
        """ Return function to compute learning rate based on the initial epoch """
        def learning_rate( step ):
            """ Compute learning rate for a given epoch, using exponential decay from lr_init to lr_final over lr_decay_steps """
            epoch = step + init_epoch            
            if epoch <= self.lr_init_steps:
                return 1.
            elif epoch >= self.lr_init_steps + self.lr_decay_steps:
                return self.lr_final/self.lr_init
            else:
                decay_factor = (self.lr_final / self.lr_init) ** ((epoch - self.lr_init_steps) / self.lr_decay_steps)
                assert 0. < decay_factor <= 1., f"Internal error: decay_factor should be in (0,1], but got {decay_factor} for epoch {epoch}"
                return decay_factor
        return learning_rate

    def unique_id(self) -> str:
        return unique_hash8(self)
    
@version("0.1", dependencies=[TrainConfig])
class Trainer(object):

    def __init__(self, 
                 label   : str|None,
                 model   : BaseModel,
                 sampler : BlockSampler,
                 config  : TrainConfig,
                 path    : SubDir|None = None,
                 restart : bool = False,
                 verbose : Context = Context.all ):
        """
        Main training loop
        """
        import numpy as nnp

        self._qual_label = f"{model.qualified_label(label)} {sampler.unique_data_id}"
        self._label      = label if ( not label is None and label != "") else model.qualified_label(None)
        self._model      = model
        self._sampler    = sampler
        self._config     = config
        self._verbose    = verbose
        self._path       = path(sampler.unique_data_id)(named_unique_filename48_8(self._label), ext="pt") if path is not None else None # data is unique, but model depends only on the label
        self._rndg       = torch.Generator(device="cpu")
        self._visualizer = model.visualizer(sampler) # might be None
        self._rndg.manual_seed(config.seed)

        self._track_val_trn = MetricTacker()
        self._track_val_vld = MetricTacker()
        self._current_epoch = 0
        self._best_epoch    = None
        self._best_val_loss = None
        self._tracker       = TrackTime()

        if verbose.shall_report():
            s = f"Trainer for '{self.qualified_label}' on data {sampler.unique_data_id}"
            verbose.write(s+'\n'+("="*len(s)))
            del s

        deleted_best = False
        if not self._path is None and restart and self._path.exists("last"):
            deleted_best = self._path.exists("best")
            self._path.delete("last")
            self._path.delete("best") if deleted_best else None
            verbose.write(lambda : f"'{self.qualified_label}': Deleted existing recovery checkpoint '{self._path.full_file_name('last')}'{' and the best-state checkpoint' if deleted_best else ''}.")

        if not self._path is None and self._path.exists("last"):
            verbose1 = verbose(1)
            with verbose1.write_t(f"Restoring checkpoint from '{self._path.full_file_name('last')}'... ", end='') as tme:
                state         = torch.load(self._path.full_file_name("last"), map_location=model.device, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                self._current_epoch    = int(state["epoch"])
                self._best_epoch    = state.get("best_epoch", None) if not deleted_best else None
                self._best_val_loss = state.get("best_val_loss", None) if not deleted_best else None

                self._track_val_trn.load_state_dict(state["track_val_trn"])
                self._track_val_vld.load_state_dict(state["track_val_vld"])

                self._optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init)
                self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=config.learning_rate(self._current_epoch))
                self._optimizer.load_state_dict(state["optimizer_state_dict"])
                self._scheduler.load_state_dict(state["scheduler_state_dict"])
                rndg_state = state["rndg_state_dict"]
                if not isinstance(rndg_state, torch.Tensor):
                    rndg_state = torch.as_tensor(rndg_state, dtype=torch.uint8)
                else:
                    rndg_state = rndg_state.detach().cpu().to(dtype=torch.uint8)
                self._rndg.set_state(rndg_state)
                verbose1.write(f"done; this took {tme}.", head=False)
        else:
            self._optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init)
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=config.learning_rate(self._current_epoch))

        verbose.report(1,lambda : f"Model is ready. Current epoch to train is {self._current_epoch}.")

    @property
    def qualified_label(self) -> str:
        """Qualified label for the model, including configuration information."""
        return self._qual_label
    
    @property
    def label(self) -> str:
        """Label for the model, as provided by the user or generated from the model class name."""
        return self._label

    @property
    def device(self):
        return self._model.device

    def __call__( self, epochs : int, *, total : bool = False ) -> BaseModel:

        verbose = self._verbose
        model   = self._model
        tracker = self._tracker
        config  = self._config
        sampler = self._sampler

        if total and self._current_epoch >= epochs:
            verbose.write(lambda : f"Model '{self.qualified_label}' was already trained for {self._current_epoch} epochs.")
            return model
        epochs = epochs if not total else ( epochs - self._current_epoch )

        verbose.write(lambda : f"Launching '{self.qualified_label}' training for {epochs} epochs over {self._sampler.num_trn_samples} training and {self._sampler.num_vld_samples} validation samples, using a mini bastch size of "+\
                               f"{config.trn_batch_size} for {config.batch_per_epoch} batches per epoch, and a batch size of {config.val_batch_size} for valuation. "+\
                               f"The model has {sum(p.numel() for p in model.parameters())} parameters." +\
                               (f" Resuming from epoch {self._current_epoch}." if self._current_epoch>0 else ""))


        # training
        # --------

        last_checkpoint_time = None

        for epoch in range(self._current_epoch+1, epochs+1):
            assert all(torch.isfinite(p).all().item() for p in model.parameters()), "Model parameters contain non-finite values before training step"
            grad_norm = 0.

            with tracker("training"):
                def train():
                    tme    = verbose.timer()
                    loss   = 0.
                    weight = 0.
                    model.train()
                    for imini in range(config.batch_per_epoch):
                        lr   = self._scheduler.get_last_lr()[0] if self._scheduler is not None else config.lr_init
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
                    self._scheduler.step() if not self._scheduler is None else None  # decay LR by 5%
                    return loss / weight, grad_norm

                trn_loss, grad_norm = train()

            lr = self._scheduler.get_last_lr()[0] if self._scheduler is not None else config.lr_init
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
                    weight  = 0.
                    metrics = {}

                    for batch in sampler.all( trn=trn, batch_size=config.val_batch_size, device=self.device ):
                        ls, wg, lm = model( batch, training=False, return_what=model.ReturnWhat.VALUATION )
                        ls      = ls.detach().item()
                        wg      = wg.detach().item()   
                        lm      = { k: v.detach().item() for k,v in lm.items() }
                        assert nnp.isfinite(ls) and nnp.isfinite(wg) and wg > 0, f"Internal error: {ls} {wg}"
                        verify( not 'loss' in lm, f"'loss' should not be returned as a separate metric, but should be included in the loss value returned by the model.")
                        loss    += ls
                        weight  += wg
                        metrics = { k: metrics.get(k, 0.) + v for k,v in lm.items() }

                    count   = metrics.pop("count")
                    assert count > 0, f"Internal error: no valid samples"
                    metrics = { k: v/count for k,v in metrics.items() }                
                    return loss / weight, metrics
                
                with tracker("val_trn"):
                    trn_val_loss, trn_metrics = value(trn=True)
                    self._track_val_trn.record( epoch, trn_val_loss, trn_metrics )
                    self._track_val_trn.record_metric( epoch, "lr", lr )
                with tracker("val_vld"):
                    vld_val_loss, vld_metrics = value(trn=False)    
                    self._track_val_vld.record( epoch, vld_val_loss, vld_metrics )
                    self._track_val_vld.record_metric( epoch, "lr", lr )

            assert not trn_val_loss is None and not vld_val_loss is None, "Internal error: valuation failed to produce a validation loss"

            # print & visualize
            # -----------------

            if verbose.shall_report():
                str_tme_val_trn = f"{tracker['val_trn'].fmt_average_lap_seconds()}"
                str_tme_val_vld = f"{tracker['val_vld'].fmt_average_lap_seconds()}"   
                str_estimated   = fmt_seconds( tracker.current_lap_seconds * (epochs - epoch) )

                str_metrics = f"LR: {lr:.2e}, GN {grad_norm:.4g}, "
                for k in trn_metrics:
                    trn_v = trn_metrics[k]
                    vld_v = vld_metrics[k]
                    str_metrics += f"{k}: {trn_v:.4g}/{vld_v:.4g}, "
                str_metrics = str_metrics[:-2] + "."
                verbose.write(f"\rTraining '{self.label}' epoch {epoch}/{epochs} error avg trn {trn_loss:.4g}; val {trn_val_loss:.4g}/{vld_val_loss:.4g} "+\
                              f"[{str_tme_trn} {str_tme_val_trn} {str_tme_val_vld}]. {str_metrics} ETA: {str_estimated}.")

            if self._visualizer is not None:
                self._visualizer.update(model,epoch=epoch, trn_tracker=self._track_val_trn, vld_tracker=self._track_val_vld) 

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
                def get_state():
                    return {
                        "epoch": int(epoch),
                        "model_state_dict": self._model.state_dict(),
                        "optimizer_state_dict": self._optimizer.state_dict(),
                        "scheduler_state_dict": None if self._scheduler is None else self._scheduler.state_dict(),
                        "rndg_state_dict": self._rndg.get_state(),
                        "track_val_trn": self._track_val_trn.state_dict(),
                        "track_val_vld": self._track_val_vld.state_dict(),
                        "best_epoch": self._best_epoch,
                        "best_val_loss": self._best_val_loss
                    }
                if not vld_val_loss is None and (self._best_val_loss is None or vld_val_loss < self._best_val_loss):
                    ffn = self._path.full_file_name("best")
                    str_prev = f" vs. {self._best_val_loss:.4g} at epoch {self._best_epoch}" if self._best_val_loss is not None else "None"
                    with verbose1.write_t(f"Epoch {epoch}: Saving new best validation loss {vld_val_loss:.4g}{str_prev} to '{ffn}'... ", end='') as tme:
                        self._best_epoch = epoch
                        self._best_val_loss = vld_val_loss
                        self._path.create_directory()
                        torch.save(get_state(), ffn)
                        verbose1.write(f"done; this took {tme}.", head=False)

                ffn = self._path.full_file_name("last")
                with verbose1.write_t(f"Saving checkpoint to {ffn}... ", end='') as tme:
                    self._path.create_directory()
                    torch.save(get_state(), ffn)
                    verbose1.write(f"done; this took {tme}.", head=False)

                last_checkpoint_time = time.time()

        return model
