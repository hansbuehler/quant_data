"""
Dysanos simple benchmark
"""
from __future__ import annotations

from quantdata.dysanos.simple.sampler import SimpleSampler
from quantdata.torch import BaseVisualizer, pop, BaseEnvironment, BaseModel, MetricTacker
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch
from dataclasses import dataclass, field
from cdxcore.err import verify_inp
from cdxcore.pretty import PrettyObject 

from enum import IntEnum
from collections.abc import Mapping
import torch as torch

@dataclass(frozen=True)
class VisualizerConfig:
    seed    : int = 1234
    num_viz : int = 4
    use_trn : bool = True

class AEC(BaseModel):
    """
    Basic AutoEncoder for Call options (AEC) model for the Dysanos simple benchmark.
    """

    VisualizerConfig = VisualizerConfig

    @dataclass(frozen=True)
    class Config:
        encode_what         : str = "vol"                                #: vol, sqrtVar, price
        num_state           : int = 32                                   #: Hidden state dimension
        num_compress        : int = 128                                  #: Internal state compression dimension for deep set
        encoder_set_config  : FeedForward.Config = FeedForward.Config()  #: Configuration for the encoder
        encoder_fix_config  : FeedForward.Config = FeedForward.Config()  #: Configuration for the encoder
        decoder_config      : FeedForward.Config = FeedForward.Config()  #: Configuration for the encoder
        vizualizer_config   : VisualizerConfig = VisualizerConfig()      #: Configuration for the visualizer

        weight_by_spread    : bool = True    # whether to weight the loss by the bid-ask spread, which puts more weight on liquid options and is a common practice in the literature; if False, all options are weighted equally
        min_pure_spread     : float = 1E-5   # floor to sprear for relative error
        mid_fit_weight      : float = 1E-3   # weight for fitting the mid price in the loss function vs exceeding bid/ask
        vol_floor           : float = 0.01   # floor for predicted volatility when using 'vol' or 'sqrtVar' 

        def __post_init__(self):
            verify_inp( self.num_state > 0, lambda : f"'num_state' must be positive, but found {self.num_state}" )
            verify_inp( self.num_compress > 0, lambda : f"'num_compress' must be positive, but found {self.num_compress}" )
            verify_inp( self.min_pure_spread > 0., lambda : f"'min_pure_spread' must be positive, but found {self.min_pure_spread:.4g}" )
            verify_inp( 0. <= self.mid_fit_weight <= 1., lambda : f"'mid_fit_weight' must be in [0,1], but found {self.mid_fit_weight:.4g}" )
            verify_inp( self.vol_floor >= 0., lambda : f"'vol_floor' must be non-negative, but found {self.vol_floor:.4g}" )

            if self.encode_what not in {"vol", "sqrtVar", "price"}:
                raise ValueError(f"Invalid value for encode_what: {self.encode_what}. Expected 'vol', 'sqrtVar' or 'price'.")           

    X_FEATURES = [
        # first element must have full dimension (B,S,O) to determine the number of options O
        'ask_pure_price',
        'bid_pure_price',
        'strike', 
        'df',
        'tte_y',
        'fwd',
        'nms',
        'ask_vol',
        'bid_vol',
    ]
    __MAY_USE = [
        #'is_american',
        #'is_am',
        'call_ask',   
        'call_bid',
        'spot_ret',
        'log_volume',
        'log_open_interest',
        'ask_pure_price',
        'bid_pure_price',
        'ask_pure_delta',
        'bid_pure_delta',
        'ask_pure_gamma',
        'bid_pure_gamma',
        'bid_pure_vega',
        'ask_pure_vega',
        'bid_pure_theta',
        'ask_pure_theta',
        'ask_pure_dk',
        'bid_pure_dk'
    ]

    CHARACTERISTICS = [
        'tte_y',
        'nms',
        'strike'
    ]

    def __init__(self, env         : BaseEnvironment, *,
                       shapes      : dict[str, tuple], # shapes of the input data fields, used for verification and to determine the number of features
                       stats       : PrettyObject, # data stats, used for verification and normalization
                       num_steps   : int, # expected number of steps
                       max_options : int, # expected maximum number of options per step
                       config      : Config = Config()
                 ):
        """ Initialize model """
        super().__init__(env=env)

        if set(shapes) != set(SimpleSampler.FEATURES):
            superfluous = set(shapes) - set(SimpleSampler.FEATURES)
            missing = set(SimpleSampler.FEATURES) - set(shapes)
            msg = "Data Input mismatch:"
            if superfluous:
                msg += f" Superfluous fields: {superfluous}."
            if missing:
                msg += f" Missing fields: {missing}."   
            raise ValueError(msg)
        
        if set(self.X_FEATURES) > set(SimpleSampler.FEATURES):
            raise RuntimeError(f"Model {self.__class__.__name__} requires features that are not provided by the data sampler: {sorted(set(self.X_FEATURES) - set(SimpleSampler.FEATURES))}.")
        if set(self.CHARACTERISTICS) > set(SimpleSampler.FEATURES):
            raise RuntimeError(f"Model {self.__class__.__name__} requires characteristics that are not provided by the data sampler: {sorted(set(self.CHARACTERISTICS) - set(SimpleSampler.FEATURES))}.")

        self.config            = config
        self.num_features      = len(self.X_FEATURES)
        self.num_characteristics = len(self.CHARACTERISTICS)
        self.num_steps         = num_steps
        self.max_options       = max_options
        self.bb                = BackboneTorch(env) # for black scholes pricer
        self.mean_pure_spread  = float( stats.mean_pure_spread )

        self.encode_set        = FeedForward( env, num_input=self.num_features, num_output=self.config.num_compress, config=self.config.encoder_set_config )
        self.encode_fix        = FeedForward( env, num_input=self.config.num_compress, num_output=self.config.num_state, config=self.config.encoder_fix_config )
        self.decode            = FeedForward( env, num_input=self.config.num_state+self.num_characteristics, num_output=0, config=self.config.decoder_config )
        self.softplus          = torch.nn.Softplus()

    def visualizer(self, sampler : SimpleSampler):
        """ Return visualizer for this model """
        return Visualizer(sampler, config = self.config.vizualizer_config )
    
    def forward(self, data          : dict, 
                      training      : bool = False,
                      return_what   : BaseModel.ReturnWhat = BaseModel.ReturnWhat.LOSS_WEIGHT ) -> tuple[torch.Tensor, torch.Tensor]|tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]|dict[str, torch.Tensor]:
        """
        Forward pass

        Parameters
        ----------
        data : dict
            Input data dictionary containing tensors for the model. The expected keys and tensor shapes depend on the specific model implementation.
        training : bool, default ``False``
            Whether the model is in training mode.
        return_what : :class:`BaseModel.ReturnWhat`, default ``ReturnWhat.LOSS_WEIGHT``
            Specifies what the model should return. Options are:
            - ``ReturnWhat.LOSS_WEIGHT``: return float ``loss`` (with gradient) and float ``weight`` (without gradient) for training.
            - ``ReturnWhat.VALUATION``: return ``loss``, ``weight`` and a dictionary of additional float metrics.
            - ``ReturnWhat.PATHWISE``: return pathwise metrics as dictionary.
        """
        assert set(data) == set(SimpleSampler.FEATURES), f"Data fields wrong: superflous: {sorted(set(data) - set(SimpleSampler.FEATURES))}, missing: {sorted(set(SimpleSampler.FEATURES) - set(data))}"
        assert not training or return_what == self.ReturnWhat.LOSS_WEIGHT, f"During training, 'return_what' must be ReturnWhat.LOSS_WEIGHT not {return_what}"

        data     = dict(data)
        mask     = pop(data,"is_valid", shape=3, dtype=torch.bool) # (B,S,O)
        B, S, O  = mask.shape
        assert S == self.num_steps, f"Expected mask to have {self.num_steps} steps as the second dimension, but got {S}"
        assert O == self.max_options, f"Expected mask to have {self.max_options} options as the third dimension, but got {O}"
        self.verify( lambda : torch.any(mask), "Batch contains no valid masked samples.")

        # encode
        # ------

        def encoder():
            features = [ data[f] for f in self.X_FEATURES ] # list of (B,S,O) tensors
            encode   = self.encode_set( features, mask = mask )
            assert encode.shape == (B, S, O, self.config.num_compress), f"Expected encoded representation to have shape {(B, S, O, self.config.num_compress)}, but found {encode.shape}"        
            encode   = torch.sum( encode, dim=2 ) / torch.sum( mask[...,None], dim=2 )# (B,S,self.config.num_compress) - average over options, weighted by mask
            encode   = torch.tanh(encode) # (B,S,self.config.num_state) - nonlinearity
            encode   = self.encode_fix( encode ) # (B,S,self.config.num_state)
            assert encode.shape == (B, S, self.config.num_state), f"Expected encoded representation after fix to have shape {(B, S, self.config.num_state)}, but found {encode.shape}"
            encode   = torch.tanh(encode) # (B,S,self.config.num_state) - nonlinearity
            return encode

        state = encoder()
        self.verify( lambda: torch.isfinite(state).all(), lambda: f"Encoded state representation contains non-finite values; found {torch.min(state[mask]).item():.4f}" )

        # project using current market
        # ----------------------------

        def decoder():
            characteristics = [ data[f] for f in self.CHARACTERISTICS ] +\
                              [ state[:,:,None,i] for i in range(state.shape[-1]) ] 
            
            for c in characteristics:
                assert len(c.shape) == 3 and c.shape[0] == B and c.shape[1] == S, f"Expected characteristic/state tensors to have shape (B,S,...) but found {c.shape}"

            decoded        = self.decode( characteristics, mask = mask ) # (B,S,O)
            assert decoded.shape == (B, S, O), f"Expected decoded representation to have shape {(B, S, O)}, but got {decoded.shape}"
            return decoded

        decoded = decoder()
        self.verify( lambda: torch.isfinite(decoded).all(), lambda: f"Decoded representation contains non-finite values; found {torch.min(decoded[mask]).item():.4f}" )

        # Compute price
        # -------------

        pure_strike = data['strike'] / torch.where( mask, data['fwd'], 1. )
        sqrtT       = data['tte_y'].sqrt() 
        pure_ask    = data['ask_pure_price']
        pure_bid    = data['bid_pure_price']
        pure_mid    = 0.5*(pure_ask + pure_bid)
        pure_spread = torch.clip( pure_ask - pure_bid, min=self.config.min_pure_spread )

        if self.config.encode_what == "vol":
            fit_vol   = self.softplus(decoded)+self.config.vol_floor # ensure positive output for volatility prediction
            fit_price = self.bb.bs_price( strikes=pure_strike, sqrtVars=fit_vol*sqrtT, mask=mask, high_precision=False, debug_verify_bounds=False )
    
        elif self.config.encode_what == "sqrtVar":
            fit_sqrtVar = self.softplus(decoded)+self.config.vol_floor*sqrtT # ensure positive output for sqrt variance prediction  
            fit_price = self.bb.bs_price( strikes=pure_strike, sqrtVars=fit_sqrtVar, mask=mask, high_precision=False, debug_verify_bounds=False )
            fit_vol   = fit_sqrtVar / torch.where( mask, sqrtT, 1. )
            del fit_sqrtVar

        else:
            assert self.config.encode_what == "price", f"Invalid value for 'encode_what': {self.config.encode_what}"
            fit_vol    = None
            fit_price  = decoded # keep a linear head so the price objective stays differentiable

        self.verify( lambda: torch.isfinite(fit_price).all(), "Infinite mid price")

        # loss function
        # -------------

        weights      = torch.where( mask, self.mean_pure_spread/torch.where(mask, pure_spread, 1.), 0. ) if self.config.weight_by_spread else\
                       torch.where( mask, 1., 0. )       
        excess       = torch.relu( fit_price - pure_ask ) + torch.relu( pure_bid - fit_price )
        loss         = weights * ( (1.-self.config.mid_fit_weight ) * excess + self.config.mid_fit_weight * ( fit_price - pure_mid )**2 )
    
        loss         = torch.sum(loss)
        with torch.no_grad():
            weight  = torch.sum(weights)

        if training or return_what == self.ReturnWhat.LOSS_WEIGHT:
            return loss, weight
        if return_what == self.ReturnWhat.VALUATION:
            with torch.no_grad():
                excess  = torch.sum(torch.where(mask,excess,0.)) if not training else None
                count   = torch.sum(mask) if not training else None
            return loss, weight, dict(excess=excess, count=count)
        
        assert return_what == self.ReturnWhat.PATHWISE, f"Invalid value for 'return_what': {return_what}"
        return dict(fit_price=torch.where( mask, fit_price, 0.),
                    fit_vol=torch.where( mask, fit_vol, 0.) if fit_vol is not None else None
                )


class Visualizer(BaseVisualizer):
    """
    Visualizer for the AEC model
    """

    Config = VisualizerConfig

    def __init__( self, sampler : SimpleSampler, config : VisualizerConfig = VisualizerConfig() ):
        from cdxcore.dynaplot import store
        import numpy as nnp

        self.sampler = sampler
        self.config  = config
        self.rng     = nnp.random.default_rng(config.seed)

        self.sample_ids  = sorted( self.rng.choice( sampler.trn_samples_ids if config.use_trn else sampler.vld_samples_ids, size=config.num_viz, replace=False ) )

        self.figStat = None
        self.figV    = None
        self.figP    = None
        self.axsV    = None
        self.axsP    = None
        self.axsF    = PrettyObject()
        self.store   = store()

    def update( self, model : AEC, *, epoch : int, trn_tracker : MetricTacker, vld_tracker : MetricTacker ):
        """ Update visualizer with new model and epoch number """
        assert isinstance(model, AEC), f"Expected model to be an instance of AEC, but got {type(model)}"
        from cdxcore.dynaplot import figure, m_o_m, colors_tableau
        from cdxcore.bs import bs
        import numpy as nnp

        model.eval()
        with torch.no_grad():
            batch         = self.sampler.generate_batch( self.sample_ids, device = model._env.device )
            pathwise      = model( batch, training=False, return_what=model.ReturnWhat.PATHWISE )

            is_valid      = batch['is_valid'].detach().cpu().numpy()
            num_opts      = is_valid.shape[-1]
            is_valid      = is_valid[:,-1,:]
            assert is_valid.shape[0] == self.config.num_viz, f"Expected number of valid samples to match number of visualizations {self.config.num_viz}, but found {is_valid.shape[0]}"

            pure_ask      = batch['ask_pure_price'][:,-1,:].detach().cpu().numpy()
            pure_bid      = batch['bid_pure_price'][:,-1,:].detach().cpu().numpy()
            bid_vols      = batch['bid_vol'][:,-1,:].detach().cpu().numpy()
            ask_vols      = batch['ask_vol'][:,-1,:].detach().cpu().numpy()
            nms           = batch['nms'][:,-1,:].detach().cpu().numpy()
            tte_y         = batch['tte_y'][:,-1,:].detach().cpu().numpy()
            fwd           = batch['fwd'][:,-1,:].detach().cpu().numpy()
            strike        = batch['strike'][:,-1,:].detach().cpu().numpy()
            pure_strike   = strike / nnp.where( is_valid, fwd, 1. ) # set pure_strike to strike for invalid samples to prevent NaNs in visualization
            sqrtT         = nnp.sqrt(tte_y)
            spread        = nnp.where( is_valid, pure_ask - pure_bid, 1. )

            fit_price     = pathwise['fit_price'][:,-1,:].detach().cpu().numpy()
            fit_vols      = pathwise['fit_vol'][:,-1,:].detach().cpu().numpy() if pathwise['fit_vol'] is not None else None

            num_plots     = fit_price.shape[0]
            is_first      = self.figV is None
            if is_first:
                self.figStat = figure( title="Training Statistics", col_size=6, row_size=4, columns=4 )
                self.axsF.loss = self.figStat.add_subplot("Losses")
                self.axsF.loss.set_yscale("log")
                self.axsF.loss.set_xlabel("Epoch")
                self.axsF.loss.set_ylabel("Log-Loss")

                self.figV = figure( title="Model vs Market Vols", col_size=6, row_size=4, columns=4 )
                self.axsV = [ self.figV.add_subplot(f"Implied Vols Sample {sid.ticker} {sid.date}") for sid in self.sample_ids ]
                for axV in self.axsV:
                    axV.set_xlabel("NMS")
                    axV.set_ylabel("Implied Volatility")

                self.figP = figure( title="Model vs Market Prices", col_size=6, row_size=4, columns=4 )
                self.axsP = [ self.figP.add_subplot(f"Call Price Differences Sample {sid.ticker} {sid.date}") for sid in self.sample_ids ]
                for axP in self.axsP:
                    axP.set_xlabel("NMS")
                    axP.set_ylabel("Price Difference")

                for axP in self.axsP:
                    axP.axhline(0., color="gray", linestyle="--", linewidth=1)

            self.store.clear()

            # convergence
            # -----------

            def progress():
                trn_loss_tracker = trn_tracker["loss"]
                trn_epochs = nnp.array(trn_loss_tracker['epochs'], dtype=nnp.float32)
                trn_losses = nnp.array(trn_loss_tracker['values'], dtype=nnp.float32)
                assert trn_epochs.shape == trn_losses.shape, f"Expected epochs and losses to have the same shape, but found {trn_epochs.shape} and {trn_losses.shape}"
                self.store += self.axsF.loss.plot( trn_epochs, trn_losses, ".-", label="Training", color="orange")
                vld_loss_tracker = vld_tracker["loss"]
                vld_epochs = nnp.array(vld_loss_tracker['epochs'], dtype=nnp.float32)
                vld_losses = nnp.array(vld_loss_tracker['values'], dtype=nnp.float32)
                assert vld_epochs.shape == vld_losses.shape, f"Expected epochs and losses to have the same shape, but found {vld_epochs.shape} and {vld_losses.shape}"
                self.store += self.axsF.loss.plot( vld_epochs, vld_losses, ".-", label="Validation", color="lightblue")

                if is_first:
                    self.axsF.loss.legend(loc="upper right")
            progress()

            # Per sample
            # ----------
            for ip, (axV, axP) in enumerate(zip(self.axsV, self.axsP)):
                ivld       = is_valid[ip,:]
                inms       = nms[ip,:][ivld]
                ixs        = nnp.argsort(inms)
                inms       = inms[ixs]
                ipure_k    = pure_strike[ip,:][ixs]
                isqrtT     = sqrtT[ip,:][ixs]
                ifit       = fit_price[ip,:][ixs]
                ibid       = pure_bid[ip,:][ixs]
                iask       = pure_ask[ip,:][ixs]
                ibid_vols  = bid_vols[ip,:][ixs]
                iask_vols  = ask_vols[ip,:][ixs]

                if not fit_vols is None:
                    ifit_vols  = fit_vols[ip,:][ixs]
                else:
                    ifit_vols  = bs.implied( ipure_k, ifit, sqrtT=isqrtT, on_exceed_bounds="quiet" )

                self.store += axV.plot( inms, ifit_vols, ".-", color="orange", label=f"Fit {ip}" )
                self.store += axV.plot( inms, ibid_vols, "^:", color="green")
                self.store += axV.plot( inms, iask_vols, "v:", color="red")

                self.store += axP.plot( inms, ibid - ifit, "^:", color="green", label=f"Bid-Fit {ip}")
                self.store += axP.plot( inms, iask - ifit, "v:", color="red", label=f"Ask-Fit {ip}")

                axV.set_ylim( *m_o_m(ifit_vols, ibid_vols, iask_vols ))
                axP.set_ylim( *m_o_m(ibid - ifit, iask - ifit) )

                if is_first:
                    axV.legend(loc="upper left")
                    axP.legend(loc="upper left")
            self.figV.suptitle( f"Model vs Market Vols Epoch {epoch}" )
            self.figP.suptitle( f"Model vs Market Prices Epoch {epoch}" )
            self.figStat.suptitle( f"Progress {epoch}" )
            self.figStat.render()
            self.figV.render()
            self.figP.render()



