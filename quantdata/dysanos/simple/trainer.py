"""
Dysanos trainer
"""

from __future__ import annotations

from quantdata.torch import get, pop, compute_grad_norm, detect_grad_issues, MetricTacker, BaseEnvironment, BaseModel
from quantdata.torch.sampler import BlockSampler
from dataclasses import dataclass
from cdxcore.err import verify_inp, verify
from cdxcore.util import TrackTime, fmt_seconds, fmt_list
from cdxcore.verbose import Context
import torch as torch

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
    
def train( env     : BaseEnvironment,
           epochs  : int,
           model   : BaseModel,
           sampler : BlockSampler,
           config  : TrainConfig,
           verbose : Context = Context.all ):
    """
    Main training loop
    """
    import numpy as nnp
    init_epoch = 0 # TODO: load state

    verbose.write(lambda : f"Launching {model.__class__.__name__} training for {epochs} epochs over {sampler.num_trn_samples} training and {sampler.num_vld_samples} validation samples, using a mini bastch size of "+\
                           f"{config.trn_batch_size} for {config.batch_per_epoch} batches per epoch, and a batch size of {config.val_batch_size} for valuation. "+\
                           f"The model has {sum(p.numel() for p in model.parameters())} parameters.")
    rndg = torch.Generator(device="cpu")

    optimizer     = torch.optim.Adam(model.parameters(), lr=config.lr_init)
    scheduler     = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=config.learning_rate(init_epoch) )

    tracker       = TrackTime()
    visualizer    = model.visualizer(sampler)
    track_val_trn = MetricTacker()
    track_val_vld = MetricTacker()

    # training
    # --------

    for epoch in range(init_epoch+1, epochs+1):

        assert all(torch.isfinite(p).all().item() for p in model.parameters()), "Model parameters contain non-finite values before training step"
        grad_norm = 0.

        with tracker("training"):
            def train():
                tme    = verbose.timer()
                loss   = 0.
                weight = 0.
                model.train()
                for imini in range(config.batch_per_epoch):
                    lr   = scheduler.get_last_lr()[0] if scheduler is not None else config.lr_init
                    verbose.write(lambda : f"\rTraining epoch {epoch}/{epochs} mini batch {imini+1}/{config.batch_per_epoch} with learning rate {lr:.2e}: so far this took {tme}...", end='')

                    batch = sampler( rndg, trn=True, batch_size=config.trn_batch_size, device=env.device, return_batch_indices=False )
                    assert batch['is_valid'].any().item(), "Sampled batch contains no valid samples. Check the sampler and the data."

                    optimizer.zero_grad(set_to_none=True)

                    assert all(torch.isfinite(p).all().item() for p in model.parameters()), f"Model parameters contain non-finite values before model call {epoch} {imini}"
                    ls, wg = model( batch, training=True )
                    mini_loss = ls / wg
                    assert torch.isfinite(mini_loss).all(), f"Internal error: mini_loss infinite"

                    mini_loss.backward()

                    detect_grad_issues(model)

                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    grad_norm = compute_grad_norm(model)

                    optimizer.step()
                    assert all(torch.isfinite(p).all().item() for p in model.parameters()), f"Model parameters contain non-finite values after optimizer step {epoch} {imini}"

                    loss    += ls.detach().item()
                    weight  += wg.detach().item()
                scheduler.step() if not scheduler is None else None  # decay LR by 5%
                return loss / weight, grad_norm

            trn_loss, grad_norm = train()

        lr = scheduler.get_last_lr()[0] if scheduler is not None else config.lr_init
        str_tme_trn = f"{tracker['training'].fmt_average_lap_seconds()}"
        verbose.write(lambda : f"\rTraining epoch {epoch}/{epochs} error avg trn {trn_loss:.4g} [{str_tme_trn}] learning rate {lr:.2e} grad norm {grad_norm:.4g}", end='')

        with torch.no_grad():
            def value( trn ):
                loss    = 0.
                weight  = 0.
                metrics = {}

                for batch in sampler.all( trn=trn, batch_size=config.val_batch_size, device=env.device ):
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
                track_val_trn.record( epoch, trn_val_loss, trn_metrics )
            with tracker("val_vld"):
                vld_val_loss, vld_metrics = value(trn=False)    
                track_val_vld.record( epoch, vld_val_loss, vld_metrics )
            
        str_tme_val_trn = f"{tracker['val_trn'].fmt_average_lap_seconds()}"
        str_tme_val_vld = f"{tracker['val_vld'].fmt_average_lap_seconds()}"   
        str_estimated   = fmt_seconds( tracker.current_lap_seconds * (epochs - epoch) )

        str_metrics = f"LR: {lr:.2e}, GN {grad_norm:.4g}, "
        for k in trn_metrics:
            trn_v = trn_metrics[k]
            vld_v = vld_metrics[k]
            str_metrics += f"{k}: {trn_v:.4g}/{vld_v:.4g}, "
        str_metrics = str_metrics[:-2] + "."
        verbose.write(lambda : f"\rTraining epoch {epoch}/{epochs} error avg trn {trn_loss:.4g}; val {trn_val_loss:.4g}/{vld_val_loss:.4g} "+\
                               f"[{str_tme_trn} {str_tme_val_trn} {str_tme_val_vld}]. {str_metrics} ETA: {str_estimated}.")
        
        visualizer.update(model,epoch=epoch, trn_tracker=track_val_trn, vld_tracker=track_val_vld)

    return model
