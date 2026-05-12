"""
Dysanos simple benchmark
"""
from __future__ import annotations
import math
import warnings

from wandb import env

from quantdata.dysanos.simple.sampler import SimpleSampler
from quantdata.pool import PoolConfig
from quantdata.torch import BaseVisualizer, pop, BaseEnvironment, BaseModel, MetricTacker, BaseSampler
from quantdata.torch.ffw import FeedForward
from quantdata.torch.bbtorch import BackboneTorch
from quantdata.biz_calendar import DAYS_PER_YEAR
from dataclasses import dataclass, field, asdict
from cdxcore.err import verify_inp
from cdxcore.uniquehash import unique_hash8
from cdxcore.pretty import PrettyObject 
from cdxcore.verbose import Context
import math as math
import torch as torch

@dataclass(frozen=True)
class VisualizerConfig:
    """ Configuration for model visualization """
    seed    : int = 1234     #: Random seed for visualization sampling, used to select a fixed set of samples for visualization across epochs
    num_viz : int = 4        #: How many samples to render
    use_trn : bool = True    #: Whether to render training or validation samples
    details : bool = False   #: Show expiry detrails
    num_tte : int = 8        #: Number of time-to-expiry bins for details
    metrics : list[str] = field(default_factory=list) #: Which metrics to plot in the training statistics figure. 

    def __post_init__(self):
        verify_inp( self.num_viz > 0, lambda : f"'num_viz' must be positive, but found {self.num_viz}" )
        verify_inp( self.num_tte > 0, lambda : f"'num_tte' must be positive, but found {self.num_tte}" )

# ------------------------------------------------------------
# Encoders
# ------------------------------------------------------------

@dataclass(frozen=True)
class EncoderFactory:
    def __call__( config, env          : BaseEnvironment,
                          num_features : int, 
                          num_state    : int ) -> callable:
        raise NotImplementedError("Encoder is an abstract class. Please implement the __call__ method.")

@dataclass(frozen=True)
class DeepSetEncoderFactory(EncoderFactory):
    num_compress : int = 128                                  #: Internal state compression dimension for deep set
    inside_tanh  : bool = True                                #: Whether to apply tanh nonlinearity in the inside of the set
    set_config   : FeedForward.Config = FeedForward.Config()  #: Configuration for the encoder
    fix_config   : FeedForward.Config = FeedForward.Config()  #: Configuration for the encoder

    def __post_init__(self):
        verify_inp( self.num_compress > 0, lambda : f"'num_compress' must be positive, but found {self.num_compress}" )

    def __call__( config, env          : BaseEnvironment,
                          num_features : int, 
                          num_state    : int ) -> callable:

        class Functional(BaseModel):

            def __init__(self):
                super().__init__(env)

                self.SET = FeedForward( env, num_input=num_features, num_output=config.num_compress, config=config.set_config )
                self.FIX = FeedForward( env, num_input=config.num_compress, num_output=num_state, config=config.fix_config )

            def forward( self, features : list[torch.Tensor], mask : torch.Tensor ) -> torch.Tensor:
                assert len(features) == num_features, f"Expected {num_features} features, but found {len(features)}"
                B, S, O = mask.shape
                C       = config.num_compress
                X       = num_state

                encode   = self.SET( features, mask = mask )
                assert encode.shape == (B, S, O, C), f"Expected encoded representation to have shape {(B, S, O, C)}, but found {encode.shape}"   
                has_opt  = torch.any( mask, dim=2 )[...,None] # should be true everywhere anyway.
                encode   = torch.where( has_opt, torch.sum( encode, dim=2 ) / torch.where(has_opt, torch.sum( mask[...,None], dim=2 ), 1.), 0. ) # (B,S,C) - average over options, weighted by mask
                encode   = torch.tanh(encode) if config.inside_tanh else encode # (B,S,C) - avoid expliding values
                encode   = self.FIX( encode ) # (B,S,X)
                assert encode.shape == (B, S, X), f"Expected encoded representation after fix to have shape {(B, S, X)}, but found {encode.shape}"
                self.verify( lambda: torch.isfinite(encode).all(), lambda: f"Encoded state representation contains non-finite values; found {torch.min(encode[mask]).item():.4f}" )
                return encode
            
        return Functional()

@dataclass(frozen=True)
class AttentionPoolingEncoderFactory(EncoderFactory):
    r"""
    AttentionPoolingEncoderFactory
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Let $f$ be a list of $n_f$ features for $m$ options and $c$ be a list of $n_c$  a list of characteristics, which are 
    non-market data features such as strike, time to expiry. The objective is to encode-decode the one of "price", "vol"
    or "sqrtVar" for each option.

    Encoding
    ^^^^^^^^

    Define:

    * $V\equiv V( f ) \in \mathbb{R}^{m, n_h}$ be the value projection from $n_f$ features to $n_h$ heads per option.
    * $Q\equiv Q( f ) \approx ( A_Q f - b_Q ) \in \mathbb{R}^{m, n_h, n_{qk}}$ be the query projection from $n_f$ features to $n_h$ queries/keys, with an inner dimension $n_{qk}$ per option.
    * $K\equiv K( f ) \approx ( A_K f - b_K ) \in \mathbb{R}^{m, n_h, n_{qk}}$ be the key projection from $n_f$ features to $n_h$ queries/keys, with an inner dimension $n_{qk}$ per option.
        
    Define the raw weights
    $$
        \alpha := \frac1{\sqrt{n_{qk}} } \exp\left( - \sum_{k=1}^{n_{qk}} Q_{o,j,k}  K_{o,j,k} \right) = \frac1{\sqrt{n_{qk}} } \exp\left( - Q_{o,j}'  K_{o,j} \right)  \in \mathbb{R}^{m, n_h} \ .
    $$
    We then define the $j$ head for $j=1,\ldots,n_h$ as
    $$
        s_j := \mathcal{N}\left( \frac{ \sum_{o=1}^m \alpha_{o,j} V_{o,j} }{  \sum_{o=1}^m  \alpha_{o,j} } \right)
    $$
    where $\mathcal{N}$ is a final network.
    """
    num_head     : int = 64    # number of heads to generate in the encoder
    num_QK       : int = 16    # projection dimension for query and key in the encoder attention mechanism.
    V_config     : FeedForward.Config = FeedForward.Config.linear_config() #: Configuration for the value projection in the self-attention mechanism
    QK_config    : FeedForward.Config = FeedForward.Config.linear_config() #: Configuration for the query projection in the self-attention mechanism
    head_config  : FeedForward.Config = FeedForward.Config()  #: Configuration for the encoder

    def __post_init__(self):
        verify_inp( self.num_head > 0, lambda : f"'num_head' must be positive, but found {self.num_head}" )
        verify_inp( self.num_QK > 0, lambda : f"'num_QK' must be positive, but found {self.num_QK}" )

    def __call__( config, env          : BaseEnvironment,
                          num_features : int, 
                          num_state    : int ) -> callable:

        class Functional(BaseModel):

            def __init__(self):
                super().__init__(env)

                self.QK   = FeedForward( env, num_input=num_features, num_output=(config.num_head,config.num_QK,2), config=config.QK_config, use_final_activation=False, use_final_bias=True )
                self.V    = FeedForward( env, num_input=num_features, num_output=config.num_head, config=config.V_config, use_final_activation=False, use_final_bias=True )
                self.head = FeedForward( env, num_input=config.num_head, num_output=num_state, config=config.head_config, use_final_activation=False, use_final_bias=True )

            def forward( self, features : list[torch.Tensor], mask : torch.Tensor ) -> torch.Tensor:
                assert len(features) == num_features, f"Expected {num_features} features, but found {len(features)}"

                B, S, O = mask.shape
                X   = num_state
                H   = config.num_head
                C   = config.num_QK   

                eQK = self.QK( features, mask = mask ) 
                eV  = self.V( features, mask = mask )
                assert eQK.shape == (B, S, O, H, C, 2), f"Expected QK representation to have shape {(B, S, O, H, C, 2)}, but found {eQK.shape}"
                assert eV.shape == (B, S, O, H), f"Expected V representation to have shape {(B, S, O, H)}, but found {eV.shape}"
                self.verify( lambda: torch.isfinite(eQK).all(), "QK is infinite")
                self.verify( lambda: torch.isfinite(eV).all(), "V is infinite")
                eQ  = eQK[...,0] # (B,S,O,H,C)
                eK  = eQK[...,1] # (B,S,O,H,C)
                eQK = None

                # unoptimized 
                scores = -(eQ * eK).sum(dim=-1) * (1.0 / math.sqrt(C))  # (B,S,O,H)

                neg_large = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~mask[..., None], neg_large)

                weight = torch.softmax(scores, dim=2)                   # normalize over O
                weight = weight.masked_fill(~mask[..., None], 0.0)

                encoded = (weight * eV).sum(dim=2)                      # (B,S,H)

                encoded = self.head( encoded, mask = None ) # (B,S,X) - head projection
                assert encoded.shape == (B, S, X), f"Expected encoded representation to have shape {(B, S, X)}, but found {encoded.shape}"
                self.verify( lambda: torch.isfinite(encoded).all(), "Encoded state is infinite")
                return encoded

        return Functional()

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass(frozen=True)
class LatentAttentionEncoderFactory(EncoderFactory):
    """
    Latent cross-attention encoder.

    For each (batch, step), a small set of learned latent queries attends to
    the option set. This gives a compressed representation directly:

        features: list[(B,S,O)]
        mask:     (B,S,O)
        output:   (B,S,num_state)

    Shapes inside attention:

        K,V:       (B*S,H,O,C)
        Q:         (B*S,H,M,C)
        attn_mask: (B*S,1,1,O), True means valid/participates
        output:    (B*S,H,M,C)

    where:
        H = num_head
        C = num_QK
        M = num_latent
    """

    num_head    : int = 8
    num_QK      : int = 16
    num_latent  : int = 8

    KV_config   : FeedForward.Config = FeedForward.Config.linear_config()
    head_config : FeedForward.Config = FeedForward.Config()

    dropout     : float = 0.0

    def __post_init__(self):
        verify_inp(self.num_head > 0, lambda: f"'num_head' must be positive, found {self.num_head}")
        verify_inp(self.num_QK > 0, lambda: f"'num_QK' must be positive, found {self.num_QK}")
        verify_inp(self.num_latent > 0, lambda: f"'num_latent' must be positive, found {self.num_latent}")
        verify_inp(0.0 <= self.dropout < 1.0, lambda: f"'dropout' must be in [0,1), found {self.dropout}")

    def __call__(
        config,
        env          : BaseEnvironment,
        num_features : int,
        num_state    : int,
    ) -> callable:

        class Functional(BaseModel):

            def __init__(self):
                super().__init__(env)

                H = config.num_head
                C = config.num_QK
                M = config.num_latent

                # Output shape: (B,S,O,2,H,C)
                # 2 = key/value.
                self.KV = FeedForward(
                    env,
                    num_input=num_features,
                    num_output=(2, H, C),
                    config=config.KV_config,
                    use_final_activation=False,
                    use_final_bias=True,
                )

                # Learned latent queries: (H,M,C)
                q = torch.empty(
                    H,
                    M,
                    C,
                    device=env.device,
                    dtype=env.fdtype,
                )
                torch.nn.init.normal_(q, mean=0.0, std=1.0 / math.sqrt(C))
                self.query = torch.nn.Parameter(q)

                # Flatten H*M*C latent-attention output to final state.
                self.head = FeedForward(
                    env,
                    num_input=H * M * C,
                    num_output=num_state,
                    config=config.head_config,
                    use_final_activation=False,
                    use_final_bias=True,
                )

            def forward(self, features: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
                assert len(features) == num_features, (
                    f"Expected {num_features} features, but found {len(features)}"
                )

                B, S, O = mask.shape
                H = config.num_head
                C = config.num_QK
                M = config.num_latent
                X = num_state

                has_opt = torch.any(mask, dim=2)  # (B,S)

                # Protect SDPA against rows where all keys are masked out.
                # For empty rows, allow option 0 to participate; KV is already zeroed
                # by FeedForward(..., mask=mask), and we zero the output afterwards.
                first_option = torch.arange(O, device=mask.device) == 0
                safe_mask = mask | ((~has_opt)[..., None] & first_option[None, None, :])

                kv = self.KV(features, mask=mask)
                assert kv.shape == (B, S, O, 2, H, C), (
                    f"Expected KV shape {(B, S, O, 2, H, C)}, found {kv.shape}"
                )

                k = kv[..., 0, :, :]  # (B,S,O,H,C)
                v = kv[..., 1, :, :]  # (B,S,O,H,C)

                self.verify(lambda: torch.isfinite(k).all(), "K is non-finite")
                self.verify(lambda: torch.isfinite(v).all(), "V is non-finite")

                # SDPA shape convention: (N,H,L,C) for query, (N,H,S,C) for key/value.
                k = k.reshape(B * S, O, H, C).transpose(1, 2)  # (B*S,H,O,C)
                v = v.reshape(B * S, O, H, C).transpose(1, 2)  # (B*S,H,O,C)

                q = self.query[None, :, :, :].expand(B * S, H, M, C)  # (B*S,H,M,C)

                # Boolean SDPA mask: True means key participates in attention.
                attn_mask = safe_mask.reshape(B * S, 1, 1, O)  # (B*S,1,1,O)

                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=(config.dropout if self.training else 0.0),
                    is_causal=False,
                )  # (B*S,H,M,C)

                y = y.transpose(1, 2).reshape(B, S, M * H * C)  # (B,S,M*H*C)

                # Remove outputs for empty rows, if any.
                y = torch.where(has_opt[..., None], y, 0.0)

                encoded = self.head(y, mask=None)
                assert encoded.shape == (B, S, X), (
                    f"Expected encoded shape {(B, S, X)}, found {encoded.shape}"
                )

                self.verify(lambda: torch.isfinite(encoded).all(), "Encoded state is non-finite")
                return encoded

        return Functional()
# ------------------------------------------------------------
# Decoders
# ------------------------------------------------------------

@dataclass(frozen=True)
class DecoderFactory:
    def __call__( config, env                 : BaseEnvironment,
                          num_characteristics : int, 
                          num_state           : int ) -> callable:
        raise NotImplementedError("Decoder is an abstract class. Please implement the __call__ method.")

@dataclass(frozen=True)
class SimpleDecoderFactory(DecoderFactory):
    config : FeedForward.Config = FeedForward.Config()  #: Configuration for the decoder

    def __call__( config, env                 : BaseEnvironment,
                          num_characteristics : int, 
                          num_state           : int ) -> callable:

        class Functional(BaseModel):

            def __init__(self):
                super().__init__(env)
                self.FF = FeedForward( env, num_input=num_characteristics+num_state, num_output=0, config=config.config )

            def forward( self, characteristics : list[torch.Tensor], state : torch.Tensor, mask : torch.Tensor ) -> torch.Tensor:
                assert len(characteristics) == num_characteristics, f"Expected {num_characteristics} characteristics, but found {len(characteristics)}"
                assert state.shape == (mask.shape[0], mask.shape[1], num_state), f"Expected state to have shape {(mask.shape[0], mask.shape[1], num_state)}, but found {state.shape}"

                B, S, O         = mask.shape
                characteristics = characteristics + [ state[:,:,None,i] for i in range(state.shape[-1]) ] 
                for c in characteristics:
                    assert len(c.shape) == 3 and c.shape[0] == B and c.shape[1] == S and c.shape[2] in [1,O], f"Expected characteristic/state tensors to have shape ({B},{S},1|{O}) but found {c.shape}"

                decoded        = self.FF( characteristics, mask = mask ) # (B,S,O)
                assert decoded.shape == (B, S, O), f"Expected decoded representation to have shape {(B, S, O)}, found {decoded.shape}"
                self.verify( lambda: torch.isfinite(decoded).all(), lambda: f"Decoded representation contains non-finite values; found {torch.min(decoded[mask]).item():.4f}" )
                return decoded

        return Functional() 

@dataclass(frozen=True)
class GatingDecoderFactory(DecoderFactory):
    num_head    : int = 16    # number of heads to generate in the decoder
    num_QK      : int = 4     # projection dimension for query and key in the decoder attention mechanism.
    V_config    : FeedForward.Config = FeedForward.Config.linear_config()       #: Configuration for the decoder value projection as multi-head output.
    QK_config   : FeedForward.Config|None = FeedForward.Config.linear_config()  #: Configuration for the decoder "attention". It can be set to None which reduces the decoder to a dense network.
    head_config : FeedForward.Config|None = FeedForward.Config()                #: Configuration for the decoder head to aggregate the heads. If None, the mean of the heads will be used.

    def __post_init__(self):
        verify_inp( self.num_head > 0, lambda : f"'num_head' must be non-negative, but found {self.num_head}" )
        verify_inp( self.num_QK >= 0, lambda : f"'num_QK' must be non-negative, but found {self.num_QK}" )

        if self.num_head == 1:
            verify_inp( self.head_config is None, lambda : f"If 'num_head' is 1, then 'head_config' must be None"   )
        if self.num_QK == 0:
            verify_inp( self.QK_config is None, lambda : f"If 'num_QK' is 0, then 'QK_config' must be None"   )
            if not self.head_config is None:
                warnings.warn(f"Specifying 'head_config' when 'num_head' in the absence of QK might be redundant")

    def __call__( config, env                 : BaseEnvironment,
                          num_characteristics : int, 
                          num_state           : int ) -> callable:

        class Functional(BaseModel):

            def __init__(self):
                super().__init__(env)
                self.QK   = FeedForward( env, num_input=num_characteristics, num_output=(config.num_head,config.num_QK,2), config=config.QK_config, use_final_activation=False, use_final_bias=True ) if config.num_QK > 0 else None
                self.V    = FeedForward( env, num_input=num_characteristics+num_state, num_output=config.num_head, config=config.V_config, use_final_activation=False, use_final_bias=True )
                self.head = FeedForward( env, num_input=config.num_head, num_output=0, config=config.head_config, use_final_activation=False, use_final_bias=True ) if config.num_head > 1 else None

            def forward( self, characteristics : list[torch.Tensor], state: torch.Tensor, mask : torch.Tensor ) -> torch.Tensor:
                assert len(characteristics) == num_characteristics, f"Expected {num_characteristics} characteristics, but found {len(characteristics)}"
                assert state.shape == (mask.shape[0], mask.shape[1], num_state), f"Expected state to have shape {(mask.shape[0], mask.shape[1], num_state)}, but found {state.shape}"

                B, S, O = mask.shape
                X       = num_state
                H       = config.num_head
                C       = config.num_QK   

                decoder_state = characteristics + [ state[:,:,None,i] for i in range(state.shape[-1]) ] 
                for c in decoder_state:
                    assert len(c.shape) == 3 and c.shape[0] == B and c.shape[1] == S and c.shape[2] in [1,O], f"Expected characteristic/state tensors to have shape ({B},{S},1|{O}) but found {c.shape}"

                dV  = self.V( decoder_state, mask = mask )
                assert dV.shape == (B, S, O, H), f"Expected V representation to have shape {(B, S, O, H)}, but found {dV.shape}"

                if self.QK is None:
                    if self.head is None:
                        decoded = torch.mean( dV, dim=3 ) # (B,S,O) - average over options
                        decoded = torch.where( mask, decoded, 0. ) # (B,S,O) - zero out non-existing options

                    else:
                        # somewhat redundant setup
                        decoded = self.head( dV, mask = mask ) # (B,S,O) - head projection
                        decoded = torch.where( mask, decoded, 0. ) # (B,S,O) - zero out non-existing options

                else:
                    dQK = self.QK( characteristics, mask = mask )
                    assert dQK.shape == (B, S, O, H, C, 2), f"Expected QK representation to have shape {(B, S, O, H, C, 2)}, but found {dQK.shape}"
                    dQ  = dQK[...,0] # (B,S,O,H,C)
                    dK  = dQK[...,1] # (B,S,O,H,C)

                    # unoptimized 
                    scores = -(dQ * dK).sum(dim=-1) * (1.0 / math.sqrt(C))  # (B,S,O,H)
                    weight = torch.softmax(scores, dim=-1)

                    if self.head is None:
                        decoded = (weight * dV).sum(dim=-1)
                        decoded = decoded.masked_fill(~mask, 0.0)
                    else:
                        decoded = self.head(dV * weight, mask=mask)

                assert decoded.shape == (B, S, O), f"Expected decoded representation to have shape {(B, S, O)}, found {decoded.shape}"
                self.verify( lambda: torch.isfinite(decoded).all(), lambda: f"Decoded representation contains non-finite values; found {torch.min(decoded[mask]).item():.4f}" )
                return decoded
            
        return Functional()

# -------------------------------------------------------
# AEC
# -------------------------------------------------------

@dataclass(frozen=True)
class AECConfig:
    """ AEC Model configuration """
    encode_what         : str = "vol"                      #: vol, sqrtVar, price
    fit_what            : str = "price"                    #: vol, sqrtVar, price
    num_state           : int = 32                         #: Hidden state dimension
    encoder             : EncoderFactory = DeepSetEncoderFactory() #: encoder factory to use
    decoder             : DecoderFactory = SimpleDecoderFactory()  #: decoder factory to use

    vol_floor           : float = 0.01   #: floor for predicted volatility when using 'vol' or 'sqrtVar' 
    fit_average         : bool = False   #: fit average per sample, rather than average accross all samples
    min_mean_spread_frac: float = 0.05   #: spread must be at least this fraction of average spread
    clip_huber_square   : bool = True    #: whether to clip the squared term in the Huber loss
    cheat               : bool = False   #: whether to cheat by giving the true volatility as input to the decoder. This is useful to isolate encoder performance from decoder performance, and should not be used in a realistic setting.

    def __post_init__(self):
        verify_inp( self.num_state > 0, lambda : f"'num_state' must be positive, but found {self.num_state}" )
        verify_inp( self.vol_floor >= 0., lambda : f"'vol_floor' must be non-negative, but found {self.vol_floor:.4g}" )
        verify_inp( 0. <= self.min_mean_spread_frac <= 1., lambda : f"'min_mean_spread_frac' must be in [0,1], but found {self.min_mean_spread_frac:.4g}" )

        if self.encode_what not in {"vol", "sqrtVar", "price"}:
            raise ValueError(f"Invalid value for encode_what: {self.encode_what}. Expected 'vol', 'sqrtVar' or 'price'.")           
        if self.fit_what not in {"vol", "sqrtVar", "price"}:
            raise ValueError(f"Invalid value for fit_what: {self.fit_what}. Expected 'vol', 'sqrtVar' or 'price'.")     
        if self.encode_what == "price":
            verify_inp( self.fit_what == "price", lambda : f"If 'encode_what' is 'price', then 'fit_what' must also be 'price', but found {self.fit_what}" )      

        verify_inp( isinstance(self.encoder, EncoderFactory), lambda : f"'encoder' must be an instance of EncoderFactory, but found {type(self.encoder)}" )
        verify_inp( isinstance(self.decoder, DecoderFactory), lambda : f"'decoder' must be an instance of DecoderFactory, but found {type(self.decoder)}" )

    def unique_id(self) -> str:
        """ Unique ID for the model configuration. Excludes 'visualizer_config' as it does not affect the model itself. """
        return unique_hash8(self)

    def instantiate_from_cache( self, env : BaseEnvironment, sampler : BaseSampler, *, verbose : Context = Context.quiet ) -> BaseModel:
        """ 
        Creates the model corresponding to this configuration, loading any necessary data from cache. This function will be called by the trainer when instantiating the model for training or evaluation. The 'verbose' argument can be used to control the verbosity of the loading process. Note that
        This function will fail if not all data are available.
        """
        return AEC( env=env, sampler=sampler, config=self, pool_config=PoolConfig.off(), override_cache_mode="cacheonly", verbose=verbose )

class AEC(BaseModel):
    """
    Basic AutoEncoder for Call options (AEC) model for the Dysanos simple benchmark.

    Parameters
    ----------
    env : :class:`BaseEnvironment`
        The environment to run the model on.
    sampler : :class:`SimpleSampler` | :class:`SimpleSampler.Config`
        The sampler to use for generating training and validation data.
        The model will not sample itself from the sampler, but it extracts
        the data shapes, training data statistics, and number of steps from the sampler to initialize itself.
        The sampler configuration is also stored in the model for future reference and caching purposes.
    config : :class:`AECConfig`, default ``AECConfig()``
        The configuration for the AEC model, including encoder and decoder configurations and other model parameters.        
    version : str, default 0.1
        The version of the model. Changing the version will invalidate any stored checkpoints
        and effectively trigger a re-generation of an experiment.
    visualizer_config : :class:`VisualizerConfig` | None, default ``None``
        Configuration for the visualizer. If set to ``None``, no visualizer will be used.
        pool_config : :class:`PoolConfig`, default ``PoolConfig.off()``
    Configuration for the data pool. This is used when the sampler is not directly instantiated with a
        sampler configuration, but rather with a sampler instance. In that case, the model needs to know how to
        configure the pool for the sampler to be able to extract the necessary data shapes and statistics from the sampler.
    """

    VisualizerConfig = VisualizerConfig

    CHARACTERISTICS = [
        'tte_y',
        'nms',
        'pure_strike'
    ]
    X_FEATURES = CHARACTERISTICS + [
        # first element must have full dimension (B,S,O) to determine the number of options O
        'mask_arb_free',
        'pure_ask_price',
        'pure_bid_price',
        'pure_fit_price',
        'ask_vol',
        'bid_vol',
        'fit_vol',
     ]
    __MAY_USE = [
        #'is_american',
        #'is_am',
        'strike',  # strike/spot_ref
        'fwd',     # fwd/spot_ref
        'df',
        'call_ask',   
        'call_bid',
        'spot_ret',
        'log_volume',
        'log_open_interest',
        'pure_ask_price',
        'pure_bid_price',
        'pure_fit_price',
        'pure_ask_delta',
        'pure_bid_delta',
        'pure_fit_delta',
        'pure_ask_gamma',
        'pure_bid_gamma',
        'pure_fit_gamma',
        'pure_bid_vega',
        'pure_ask_vega',
        'pure_fit_vega',
        'pure_bid_theta',
        'pure_ask_theta',
        'pure_fit_theta',
        'pure_ask_dk',
        'pure_bid_dk'
        'pure_fit_dk',
    ]

    Config = AECConfig

    def __init__(self, env               : BaseEnvironment, *,
                       sampler           : SimpleSampler|SimpleSampler.Config,
                       config            : Config = Config(),
                       version           : str = "0.1",                   #: change in version will re-generate an experiment
                       visualizer_config : VisualizerConfig|None = None,  #: Configuration for the visualizer. Set to ``None`` for no visualizer.
                       pool_config       : PoolConfig = PoolConfig.off()
                 ):
        """ Initialize model """
        super().__init__(env=env)

        if isinstance(sampler, SimpleSampler.Config):
            sampler = SimpleSampler(sampler,pool_config=pool_config)

        shapes      = sampler.shapes()
        stats       = sampler.stats()
        num_steps   = sampler.num_steps
        max_options = sampler.max_options
        
        set_f       = set(sampler.features)
        if set(shapes) != set_f:
            superfluous = set(shapes) - set_f
            missing     = set_f - set(shapes)
            if missing:
                msg = f" [Superfluous fields: {sorted(superfluous)}]" if superfluous else ""
                raise ValueError(f"Missing fields in source data: {sorted(missing)}{msg}")
            warnings.warn(f"Data set contains more data than required: {sorted(superfluous)}.")

        set_x = set(self.X_FEATURES)
        set_c = set(self.CHARACTERISTICS)
        if set_x > set_f:
            raise RuntimeError(f"Model {self.__class__.__name__} requires features that are not provided by the data sampler: {sorted(set_x - set_f)}.")
        if set_c > set_f:
            raise RuntimeError(f"Model {self.__class__.__name__} requires characteristics that are not provided by the data sampler: {sorted(set_c - set_f)}.")
        assert len(self.X_FEATURES) == len(set_x), f"Duplicate features in X_FEATURES: {self.X_FEATURES}"
        assert len(self.CHARACTERISTICS) == len(set_c), f"Duplicate features in CHARACTERISTICS: {self.CHARACTERISTICS}"

        self.version           = str(version)
        self.config            = config
        self.visualizer_config = visualizer_config
        self.sampler_config    = sampler.config
        self.num_features      = len(self.X_FEATURES)
        self.num_characteristics = len(self.CHARACTERISTICS)
        self.num_steps         = num_steps
        self.max_options       = max_options
        self.bb                = BackboneTorch(env) # for black scholes pricer
        self.stats             = stats
        self.sampler_features  = sampler.features

        self.encoder           = config.encoder( env=env, num_features=self.num_features, num_state=config.num_state )
        self.decoder           = config.decoder( env=env, num_characteristics=(self.num_characteristics+1) if config.cheat else self.num_characteristics, num_state=config.num_state )
        self.softplus          = torch.nn.Softplus()

    def visualizer(self, sampler : SimpleSampler):
        """ Return visualizer for this model """
        return Visualizer(sampler, config = self.visualizer_config ) if not self.visualizer_config is None else None
    
    @property
    def default_label(self) -> str:
        """
        Returns a default label which is the model name with a unique ID for the configuration.
        Usually that's too restrictive as it doesn not allow small changes in model parameters
        in the sense that those will generate a new experiment
        """
        return f"{self.__class__.__name__} {self.config.unique_id()}"
    
    def info_qualified_label(self, label : str|None = None) -> str:
        """
        Returns a label which uniquely identified the model based on its configuration and training data.
        This is great for plotting user information.
        Do not store information with this label as this label is often too harsh.
        Instead, rely on the user to alter "label".
        """
        return f"{self.__class__.__name__} {label} {self.version} [{self.config.unique_id()}|{self.sampler_config.unique_data_id()}]"

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
        might_pad : bool, default ``True``
            Whether the input data might contain padding samples. In this case, handle the boolean
            ``pad_mask`` tensor of shape ``(batch_size, 1,1)``.
        return_what : :class:`BaseModel.ReturnWhat`, default ``ReturnWhat.LOSS_WEIGHT``
            Specifies what the model should return. Options are:
            - ``ReturnWhat.LOSS_WEIGHT``: return float ``loss`` (with gradient) and float ``weight`` (without gradient) for training.
            - ``ReturnWhat.VALUATION``: return ``loss``, ``weight`` and a dictionary of additional float metrics.
            - ``ReturnWhat.PATHWISE``: return pathwise metrics as dictionary.
        """
        data     = dict(data)
        pad_mask = data.pop('pad_mask',None)
        assert set(data) <= set(self.sampler_features), f"Missing data fields: {sorted(set(data) - set(self.sampler_features))}"
        assert not training or return_what == self.ReturnWhat.LOSS_WEIGHT, f"During training, 'return_what' must be ReturnWhat.LOSS_WEIGHT not {return_what}"

        mask     = pop(data,"is_valid", shape=3, dtype=torch.bool) # (B,S,O)
        B, S, O  = mask.shape
        X        = self.config.num_state
        assert S == self.num_steps, f"Expected mask to have {self.num_steps} steps as the second dimension, found {S}"
        assert O == self.max_options, f"Expected mask to have {self.max_options} options as the third dimension, found {O}"
        self.verify( lambda : torch.any(mask), "Batch contains no valid masked samples.")
        assert not training or (pad_mask is None), f"'pad_mask' must not be provided during training"
        assert pad_mask is None or (pad_mask.shape == (B, 1, 1) and pad_mask.dtype == torch.bool), f"'pad_mask' must have shape {(B, 1, 1)} and dtype torch.bool, but found {pad_mask.shape} and {pad_mask.dtype}"

        # encode
        # ------
    
        features = [ data[f] for f in self.X_FEATURES ] # list of (B,S,O) tensors
        characteristics = [ data[f] for f in self.CHARACTERISTICS ]

        if self.config.cheat:
            characteristics = characteristics + [ data['fit_vol'] ]

        encoded = self.encoder( features, mask=mask if pad_mask is None else (mask&pad_mask ) ) # (B,S,X)
        assert encoded.shape == (B, S, X), f"Expected encoded representation to have shape {(B, S, X)}, found {encoded.shape}"
        self.verify( lambda: torch.isfinite(encoded).all(), lambda: f"Encoded representation contains non-finite values")
        encoded = torch.tanh( encoded )

        decoded = self.decoder( characteristics, encoded, mask=mask if pad_mask is None else (mask&pad_mask )  ) # (B,S,O)
        assert decoded.shape == (B, S, O), f"Expected decoded representation to have shape {(B, S, O)}, found {decoded.shape}"
        self.verify( lambda: torch.isfinite(decoded).all(), lambda: f"Decoded representation contains non-finite values")
        encoded = None

        # Compute price
        # -------------

        pure_strike   = data['pure_strike']
        sqrtT         = data['tte_y'].sqrt() 
        pure_ask      = data['pure_ask_price']
        pure_bid      = data['pure_bid_price']
        ask_vol       = data['ask_vol']
        bid_vol       = data['bid_vol']
        mask_arb_free = data['mask_arb_free']

        if self.config.encode_what == "vol":
            fit_vol      = self.softplus(decoded)+self.config.vol_floor # ensure positive output for volatility prediction
            fit_sqrtVar  = fit_vol*sqrtT
            fit_price    = self.bb.bs_price( strikes=pure_strike, sqrtVars=fit_sqrtVar, mask=mask, high_precision=False, debug_verify_bounds=False )
    
        elif self.config.encode_what == "sqrtVar":
            fit_sqrtVar = self.softplus(decoded)+self.config.vol_floor*sqrtT # ensure positive output for sqrt variance prediction  
            fit_price   = self.bb.bs_price( strikes=pure_strike, sqrtVars=fit_sqrtVar, mask=mask, high_precision=False, debug_verify_bounds=False )
            fit_vol     = fit_sqrtVar / torch.where( mask, sqrtT, 1. )

        else:
            assert self.config.encode_what == "price", f"Invalid value for 'encode_what': {self.config.encode_what}"
            fit_price   = self.softplus(decoded) # keep positivity
            fit_vol     = None
            fit_sqrtVar = None
        decoded = None

        self.verify( lambda: torch.isfinite(fit_price).all(), "Infinite mid price")

        if return_what == self.ReturnWhat.PATHWISE:
            if not pad_mask is None:
                mask = mask & pad_mask
            return dict(fit_price=torch.where( mask, fit_price, 0.),
                        fit_vol=torch.where( mask, fit_vol, 0.) if fit_vol is not None else None
                    )

        # loss function
        # -------------

        def loss_f( fit, bid, ask, mean_spread ):
            """
            Computes a weighted loss.
            In theory we want to express the loss per option in relation to bid/ask.
            However, this is too large for initial values hence we scale it back to
            average spread. 

            'pad_mask' will be None when training but a valid mask otherwise.
            Note that padded samples are guaranteed to be valid, including
            having at least one valid option.
            """
            half_spread  = torch.where( mask, torch.clip( ask - bid, min=self.config.min_mean_spread_frac * mean_spread ), 1. ) / 2.
            mid          = (bid + ask) / 2.
            x            = torch.where(mask, (fit - mid) / half_spread, 0.0)
            excess       = torch.relu(x.abs() - 1.0)  # excess is 0 between bid and ask, and grows linearly outside of the bid-ask
            xsq          = torch.clip( x**2, max=1.) if self.config.clip_huber_square else (x**2) # huber loss for better stability at the beginning of training; quadratic loss with flat minimum between bid and ask; value is 0 at the mid, 1 at the bid and ask, and grows quadratically outside of the bid-ask
            loss         = xsq + excess # quadratic loss with flat minimum between bid and ask; value is 0 at the mid, 1 at the bid and ask, and grows quadratically outside of the bid-ask

            if not self.config.fit_average:
                with torch.no_grad():
                    loss_sq = loss**2
                    loss_sq = loss_sq if pad_mask is None else (loss_sq*pad_mask)
                    loss_sq  = torch.sum(loss_sq)
                    weights  = torch.sum(mask if pad_mask is None else (mask & pad_mask) )
                loss     = torch.sum(loss if pad_mask is None else (loss*pad_mask))

            else:
                mask2 =  torch.sum(mask, dim=2)
                self.verify( lambda :mask2.min() > 0, "Internal error - all samples must have some options")
                with torch.no_grad():
                    loss_sq = torch.sum( loss**2, dim=2 ) / mask2 
                    loss_sq = loss_sq if pad_mask is None else (loss_sq*pad_mask[...,0])
                    loss_sq = torch.sum( loss_sq )  # we cannot
                    wgts_2  = torch.any(mask, dim=2)
                    wfts_2  = wgts_2 if pad_mask is None else (wgts_2 & pad_mask[...,0])
                    weights = torch.sum( wfts_2 )
                loss    = torch.sum( loss, dim=2 ) / mask2
                loss    = loss if pad_mask is None else (loss*pad_mask[...,0])
                loss    = torch.sum( loss )  # we cannot
            return loss, loss_sq, weights
        
        if self.config.fit_what == "price":
            loss, loss_sq, weights = loss_f( fit_price, pure_bid, pure_ask, self.stats.mean_pure_spread )

        elif self.config.fit_what == "vol":
            loss, loss_sq, weights = loss_f( fit_vol, bid_vol, ask_vol, self.stats.mean_vol_spread )
    
        else:
            assert self.config.fit_what == "sqrtVar", f"Invalid value for 'fit_what': {self.config.fit_what}"
            loss, loss_sq, weights = loss_f( fit_sqrtVar, bid_vol*sqrtT, ask_vol*sqrtT, self.stats.mean_sqrtVar_spread )

        if training or return_what == self.ReturnWhat.LOSS_WEIGHT:
            return loss, weights

        # return metrics
        # --------------

        assert not pad_mask is None, "Internal error - pad_mask must be provided to return metrics"
        assert return_what == self.ReturnWhat.VALUATION, f"Invalid value for 'return_what': {return_what}"
        mask     = mask & pad_mask # (B,S,O) - valid options that are not padded out
        pad_mask = None

        with torch.no_grad():
            # to get an understandable excess in relation to the spread
            # note that 'mask' is now padded, hence some samples may not
            # have any valid option.
            excess            = torch.where( mask, torch.clip(fit_price - pure_ask, min=0.) + torch.clip(pure_bid - fit_price, min=0.), 0. ) # how much we are outside bid/ask in $
            half_spread       = torch.where( mask, 0.5* torch.clip( pure_ask - pure_bid, min=self.config.min_mean_spread_frac * self.stats.mean_pure_spread ), 1. )
            inv_hspread       = torch.where( mask, 1./half_spread, 0. )
            rel_excess        = excess*inv_hspread # how much are we outside the bid/ask relative to half spreadf
            rel_excess_sq     = rel_excess**2
            excess_sq         = excess**2
            sum_dim_mask2     = torch.sum( mask, dim=2 ) # will be zero for padded samples
            has_dim2          = sum_dim_mask2 > 0
            inv_sum_dim_mask2 = torch.where( has_dim2, 1./torch.where( has_dim2, sum_dim_mask2, 1.), 0. )
            inv_ispread_mask2 = torch.where( has_dim2, 1./torch.where( has_dim2, torch.sum(inv_hspread, dim=2), 1.), 0. )
            sum_any_mask      = torch.sum( torch.any(mask, dim=2) ) # cannot be zero since at least some samples are not padded out
            sum_mask          = torch.sum( mask ) # cannot be zero since at least some samples are not padded out
            self.verify( lambda : torch.all(sum_any_mask > 0), "Found non-padded samples with zero options")
            has_dim2 = None
            half_spread = None

            metrics = dict(
                # excess, weighted by spread, then normalized.
                excess_weighted =     ( torch.sum(inv_hspread*excess),
                                        torch.sum(inv_hspread*excess_sq),
                                        torch.sum(inv_hspread) ),
                excess_weighted_avg = ( torch.sum( torch.sum(inv_hspread*excess, dim=2 ) * inv_ispread_mask2 ),
                                        torch.sum( torch.sum(inv_hspread*excess_sq, dim=2 ) * inv_ispread_mask2 ),
                                        sum_any_mask ),
                # excess weighted by spread
                excess_spread   =     ( torch.sum(rel_excess),
                                        torch.sum(rel_excess_sq),
                                        sum_mask ),         
                excess_spread_avg =   ( torch.sum( torch.sum(rel_excess, dim=2 ) * inv_sum_dim_mask2 ),
                                        torch.sum( torch.sum(rel_excess_sq, dim=2 ) * inv_sum_dim_mask2 ),
                                        sum_any_mask ),
                # raw violation in $
                excess =              ( torch.sum( excess ), 
                                        torch.sum( excess_sq ), 
                                        sum_mask ),
                excess_avg =          ( torch.sum( torch.sum( excess, dim=2 ) * inv_sum_dim_mask2 ),
                                        torch.sum( torch.sum( excess_sq, dim=2 ) * inv_sum_dim_mask2 ),
                                        sum_any_mask )
            )

        return loss, loss_sq, weights, metrics

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

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
        self.details = None
        self.axsV    = None
        self.axsP    = None
        self.axsF    = PrettyObject()
        self.axsF2   = PrettyObject()
        self.store   = store()
        self.metrics = self.config.metrics if 'loss' in self.config.metrics else ( ['loss'] + self.config.metrics )

    def update( self, model : AEC, *, epoch : int, trn_tracker : MetricTacker, vld_tracker : MetricTacker ):
        """ Update visualizer with new model and epoch number """
        assert isinstance(model, AEC), f"Expected model to be an instance of AEC, found {type(model)}"
        from cdxcore.dynaplot import figure, m_o_m, quantile_m_o_m
        from cdxcore.bs import bs
        import numpy as nnp

        is_first      = self.figV is None
        self.store.clear()
        import matplotlib.pyplot as plt

        _SMALL_FONTS = {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
        }
        with plt.rc_context(_SMALL_FONTS):

            # convergence
            # -----------

            def progress():
                trn_lr_tracker = trn_tracker.get("lr", None)
                if trn_lr_tracker is not None:
                    trn_epochs = nnp.array(trn_lr_tracker['epochs'], dtype=nnp.float32)
                    trn_lr = nnp.array(trn_lr_tracker['values'], dtype=nnp.float32)

                if is_first:
                    self.figStat = figure( title="Training Statistics", col_size=4, row_size=4, columns=None )
                    for metric in self.metrics:
                        ax = self.figStat.add_subplot(metric)
                        ax.set_yscale("log")
                        ax.set_xlabel("Epoch")
                        self.axsF[metric] = ax
                        if not trn_lr_tracker is None and metric == "loss":
                            ax2 = ax.twinx()
                            ax2.set_ylabel("Learning Rate")
                            self.axsF2[metric] = ax2


                def add_trn_vld_metric(ax, tracker, metric : str, label:str, color:int):
                    if tracker is None:
                        return None
                    epochs = nnp.array(tracker['epochs'], dtype=nnp.float32)
                    values = nnp.array(tracker['values'], dtype=nnp.float32)
                    stds   = nnp.array(tracker['stds'], dtype=nnp.float32) if tracker['stds'] is not None else None
                    assert epochs.shape == values.shape, f"'{metric}' '{label}': epochs and values to have the same shape, but found {epochs.shape} and {values.shape}"
                    if len(values) == 0:
                        return None
                    self.store += ax.plot( epochs, values, ".-", label=label, color=color)
                    if stds is not None:
                        rx = nnp.exp( -stds/values )  # ~ 1-stds
                        self.store += ax.fill_between(epochs, values * rx, values / rx, color=color, alpha=0.1)
                    return values, stds

                def add_metric(metric):
                    ax = self.axsF[metric]
                    trn_v, trn_std = add_trn_vld_metric( ax, trn_tracker.get(metric, None), metric, "Training", color="orange" )
                    vld_v, vld_std = add_trn_vld_metric( ax, vld_tracker.get(metric, None), metric, "Validation", color="lightblue" )
                    if is_first:
                        ax.legend(loc="upper right")
                    if not trn_v is None and not vld_v is None:
                        ax.set_ylim( quantile_m_o_m(trn_v, vld_v, hi=0.9, lo=0., pos_floor=0.1, buf=0.01) )
                        ax.set_title( f"{metric}\n{nnp.mean(trn_v[-10:]):.2g} [{nnp.mean(trn_std[-10:]):.2g}] / {nnp.mean(vld_v[-10:]):.2g} [{nnp.mean(vld_std[-10:]):.2g}]" )
                    
                    if not trn_lr_tracker is None and metric == "loss":
                        ax2 = self.axsF2.get(metric, None)
                        self.store += ax2.plot( trn_epochs, trn_lr, "--", color="green")

                # loss
                # others                

                for metric in self.metrics:
                    add_metric(metric)

                self.figStat.suptitle( f"Progress {epoch}" )
                self.figStat.render()
            progress()  

            # model evaluation
            # ----------------

            model.eval()
            with torch.no_grad():
                batch         = self.sampler.generate_batch( self.sample_ids, device = model._env.device )
                pathwise      = model( batch, training=False, return_what=model.ReturnWhat.PATHWISE )

            is_valid      = batch['is_valid'].detach().cpu().numpy()
            is_valid      = is_valid[:,-1,:]
            assert is_valid.shape[0] == self.config.num_viz, f"Expected number of valid samples to match number of visualizations {self.config.num_viz}, but found {is_valid.shape[0]}"

            pure_ask      = batch['pure_ask_price'][:,-1,:].detach().cpu().numpy()
            pure_bid      = batch['pure_bid_price'][:,-1,:].detach().cpu().numpy()
            bid_vols      = batch['bid_vol'][:,-1,:].detach().cpu().numpy()
            ask_vols      = batch['ask_vol'][:,-1,:].detach().cpu().numpy()
            nms           = batch['nms'][:,-1,:].detach().cpu().numpy()
            tte_y         = batch['tte_y'][:,-1,:].detach().cpu().numpy()
            pure_strike   = batch['pure_strike'][:,-1,:].detach().cpu().numpy()
            sqrtT         = nnp.sqrt(tte_y)

            fit_price     = pathwise['fit_price'][:,-1,:].detach().cpu().numpy()
            fit_vols      = pathwise['fit_vol'][:,-1,:].detach().cpu().numpy() if pathwise['fit_vol'] is not None else None
            fit_vols      = bs.implied( pure_strike, fit_price, sqrtT=sqrtT, mask=is_valid, on_exceed_bounds="quiet" ) if fit_vols is None else fit_vols
            batch = None
            pathwise = None 

            # per sample, aggregate price and vol error
            # ------------------------------------------

            def per_sample():
                if is_first:
                    self.figV = figure( title="Model vs Market Vols", col_size=6, row_size=4, columns=None )
                    self.axsV = [ self.figV.add_subplot(f"Implied Vols Sample {sid.ticker} {sid.date}") for sid in self.sample_ids ]
                    for axV in self.axsV:
                        axV.set_xlabel("Ask Vol")
                        axV.set_ylabel("Implied Volatility")

                    self.figP = figure( title="Model vs Market Prices", col_size=6, row_size=4, columns=None )
                    self.axsP = [ self.figP.add_subplot(f"Call Price Differences Sample {sid.ticker} {sid.date}") for sid in self.sample_ids ]
                    for axP in self.axsP:
                        axP.set_xlabel("NMS")
                        axP.set_ylabel("Price Difference")
                    for axP in self.axsP:
                        axP.axhline(0., color="gray", linestyle="--", linewidth=1)

                for ip, (axV, axP) in enumerate(zip(self.axsV, self.axsP)):
                    ivld       = is_valid[ip,:]
                    ipure_k    = pure_strike[ip,:][ivld]
                    inms       = nms[ip,:][ivld]
                    isqrtT     = sqrtT[ip,:][ivld]
                    ifit       = fit_price[ip,:][ivld]
                    ibid       = pure_bid[ip,:][ivld]
                    iask       = pure_ask[ip,:][ivld]
                    ibid_vols  = bid_vols[ip,:][ivld]
                    iask_vols  = ask_vols[ip,:][ivld]
                    ifit_vols  = fit_vols[ip,:][ivld]

                    ixs        = nnp.argsort(iask_vols)
                    ibid_vols_ = ibid_vols[ixs]
                    iask_vols_ = iask_vols[ixs]
                    ifit_vols_ = ifit_vols[ixs]

                    self.store += axV.plot( iask_vols_, ifit_vols_, ".-", color="orange", label=f"Fit {ip}" )
                    self.store += axV.plot( iask_vols_, ibid_vols_, "^:", color="green")
                    self.store += axV.plot( iask_vols_, iask_vols_, "v:", color="red")

                    ixs        = nnp.argsort(inms)
                    inms_      = inms[ixs]
                    ifit_      = ifit[ixs]
                    ibid_      = ibid[ixs]
                    iask_      = iask[ixs]
                    self.store += axP.plot( iask_vols_, ibid_ - ifit_, "^:", color="green", label=f"Bid-Fit {ip}")
                    self.store += axP.plot( iask_vols_, iask_ - ifit_, "v:", color="red", label=f"Ask-Fit {ip}")
                    if is_first:
                        ispread_ = iask_ - ibid_
                        assert nnp.min(ispread_) > 0., f"Negative spread found in sample {ip}: min spread is {nnp.min(ispread_):.4g}"
                        axP.fill_between( iask_vols_, -ispread_, +ispread_, color="blue", alpha=0.2, label="Spread" )

                    axV.set_ylim( *m_o_m(ifit_vols, ibid_vols, iask_vols ))
                    axP.set_ylim( *m_o_m(ibid - ifit, iask - ifit) )

                    if is_first:
                        axV.legend(loc="upper left")
                        axP.legend(loc="upper left")
                self.figV.suptitle( f"Model vs Market Vols Epoch {epoch}" )
                self.figP.suptitle( f"Model vs Market Prices Epoch {epoch}" )
                self.figV.render()
                self.figP.render()
            per_sample()

            # details
            # -------

            def details():
                if not self.config.details:
                    return
                
                if is_first:
                    def create_detail_fig(sid):
                        r = PrettyObject()
                        r.fig = figure( title=f"Model Details {sid.ticker} {sid.date}", col_size=3, row_size=3, columns=None )
                        r.axs = []
                        for i in range(self.config.num_tte):
                            ax = r.fig.add_subplot(f"TTE #{i}")
                            ax.set_xlabel("NMS")
                            ax.set_ylabel("Vol")
                            r.axs.append(ax)
                        return r
                    self.details = [ create_detail_fig(sid) for sid in self.sample_ids ]

                for ip, siddata in enumerate(self.details):
                    sid = self.sample_ids[ip]

                    xis_valid = is_valid[ip,:]
                    xnms     = nms[ip,:][xis_valid]
                    ixs      = nnp.argsort(xnms)
                    xnms     = xnms[ixs]
                    xtte_y   = tte_y[ip,:][xis_valid][ixs]
                    xask_vols = ask_vols[ip,:][xis_valid][ixs]
                    xbid_vols = bid_vols[ip,:][xis_valid][ixs]
                    xfit_vols = fit_vols[ip,:][xis_valid][ixs]

                    uttes = nnp.unique(xtte_y).tolist()
                    uttes = uttes if len(uttes) <= self.config.num_tte else ( uttes[:self.config.num_tte-2] + uttes[-2:] ) # limit number of TTEs visualized in deta

                    for tte, ax in zip(uttes, siddata.axs[:len(uttes)]):
                        mask = xtte_y == tte

                        xav = xask_vols[mask]
                        xbv = xbid_vols[mask]
                        xfv = xfit_vols[mask]
                        xnm = xnms[mask]

                        self.store += ax.plot( xnm, xav, ".:", color="red", label="Ask Vol" )
                        self.store += ax.plot( xnm, xbv, ".:", color="green", label="Bid Vol" )
                        self.store += ax.plot( xnm, xfv, ".-", color="orange", label="Fit Vol" )

                        ax.set_ylim( m_o_m(xav,xbv,xfv) )
                        ax.set_title( f"TTE {tte*DAYS_PER_YEAR:.2f}d" )

                    siddata.fig.render()
            details()

                



