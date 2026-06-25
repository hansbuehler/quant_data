"""
Static Model Visualization
"""

import numpy as np
from cdxcore.dynaplot import FigStore, colors_tableau, color_tableau, figure, m_o_m, quantile_m_o_m
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir
from dataclasses import dataclass

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_model import SanosModelConfig

@dataclass(frozen=True)
class VizConfig:
    num_viz_dates : int|list[int] = 4       # how many implied vols to render
    seed          : int = 1231231 # seed for chosing the above
    use_nms       : bool = False   # whether use NMS coordinates or K

    def __post_init__(self):
        if isinstance(self.num_viz_dates, int):
            assert self.num_viz_dates > 0, f"Expected 'num_viz_dates' {self.num_viz_dates} to be greater than 0"
        elif isinstance(self.num_viz_dates, list):
            assert all(n >= 0 for n in self.num_viz_dates), f"Expected all 'num_viz_dates' {self.num_viz_dates} to be greater than 0"
        else:
            raise ValueError(f"Expected 'num_viz_dates' to be int or list of int, but found {type(self.num_viz_dates)}")
        assert self.seed >= 0, f"Expected 'seed' {self.seed} to be greater than or equal to 0"

class Visualizer(object):

    class ExpiryGfx(object):

        def __init__(self, ticker, *, 
                           date, index, market, model_output0,
                           viz_config : VizConfig,
                           mdl_config : SanosModelConfig ):
            self.fig        = figure(f"{ticker} #{index} {date}", col_size=5, row_size=5, columns=None )
            self.index      = index
            self.date       = date
            self.market     = market
            self.viz_config = viz_config    
            self.mdl_config = mdl_config
            self.store      = FigStore()
            self.first      = True
            assert np.max(self.index) < market['expiries'].shape[0], f"Expected index {self.index} to be less than expiries.shape[0] {market['expiries'].shape[0]}"

            # market coordinates
            self.expiries    = market['expiries'][self.index,:]
            self.mkt_strikes = market['xstrikes'][self.index,:,1:-1]
            self.mkt_nms     = market['nms'][self.index,:,:]
            self.mids        = market['mids'][self.index,:,:]
            self.mid_vols    = market['mid_vols'][self.index,:,:]
            self.weights     = market['weights'][self.index,:,:]
            self.mkt_fig_strikes = self.mkt_nms if self.viz_config.use_nms else self.mkt_strikes
            strk_name        = "NMS" if self.viz_config.use_nms else "K"
            self.mkt_exp_block = np.stack( [ self.expiries*DAYS_PER_YEAR ]*(self.mkt_strikes.shape[-1]), axis=-1 )
            exp_range = (self.expiries.max()*DAYS_PER_YEAR, self.expiries.min()*DAYS_PER_YEAR)

            # market
            # ------

            assert self.mkt_exp_block.shape == self.mkt_nms.shape == self.mid_vols.shape, f"Expected mkt_exp_block, nms and mid_vols to have the same shape, but found {self.mkt_exp_block.shape}, {self.mkt_nms.shape} and {self.mid_vols.shape}"
            self.axMarketVols = self.fig.add_subplot("Interpolated Market Vols", projection="3d") 
            self.axMarketVols.plot_surface( self.mkt_exp_block, self.mkt_fig_strikes, self.mid_vols, cmap='viridis', edgecolor='none')
            self.axMarketVols.set_xlabel("Days")
            self.axMarketVols.set_ylabel(strk_name)
            self.axMarketVols.view_init(elev=25, azim=45) 
            self.axMarketVols.set_xlim(*exp_range)

            self.axFittedVols = self.fig.add_subplot("Model Vols", projection="3d")
            self.axFittedVols.set_xlabel("Days")
            self.axFittedVols.set_ylabel(strk_name)
            self.axFittedVols.view_init(elev=25, azim=45) 
            self.axFittedVols.set_xlim(*exp_range)

            self.axAbsVolsDiff = self.fig.add_subplot("Abs Vols Difference", projection="3d")
            self.axAbsVolsDiff.set_xlabel("Days")
            self.axAbsVolsDiff.set_ylabel(strk_name)
            self.axAbsVolsDiff.set_zlim( 0., 0.05 )
            self.axAbsVolsDiff.view_init(elev=25, azim=45) 
            self.axAbsVolsDiff.set_xlim(*exp_range)

            self.axWeightedAbsVolsDiff = self.fig.add_subplot("Weighted Abs Vols Difference", projection="3d")
            self.axWeightedAbsVolsDiff.set_xlabel("Days")
            self.axWeightedAbsVolsDiff.set_ylabel(strk_name)
            self.axWeightedAbsVolsDiff.set_zlim( 0., 0.05 )
            self.axWeightedAbsVolsDiff.view_init(elev=25, azim=45) 
            self.axWeightedAbsVolsDiff.set_xlim(*exp_range)

            # Model
            # -----

            self.axDLVs = self.fig.add_subplot("DLVs", projection="3d")
            self.axDLVs.set_xlabel("Days")
            self.axDLVs.set_ylabel(strk_name)
            self.axDLVs.set_zlabel("DLV")
            self.axDLVs.view_init(elev=25, azim=45) 
            self.axDLVs.set_xlim(*exp_range)

            self.axDensity = self.fig.add_subplot("Density", projection="3d")
            self.axDensity.set_xlabel("Days")
            self.axDensity.set_ylabel(strk_name)
            self.axDensity.set_zlabel("Density")
            self.axDensity.view_init(elev=25, azim=45)
            self.axDensity.set_xlim(*exp_range)

            # fwd vols
            expiries       = self.market['expiries'][self.index,:]
            variances      = self.market['variances'][self.index,:]
            vols           = np.sqrt( np.diff(variances,prepend=0.)/np.diff(expiries, prepend=0.) )
            self.axFwdVols = self.fig.add_subplot("Fwd Vols")
            self.axFwdVols.plot( np.sqrt(expiries), vols, ".:", color="blue", label="Market")
            self.axFwdVols.set_xlabel("Sqrt(T)")

            # state
            self.axState = self.fig.add_subplot("State")

            self.fig.render()
            self.update( epoch=0, model_output=model_output0, final=False )

        def update(self, *, epoch, model_output, final : bool = False ):
            """ update this date """

            # market coordinates
            # ------------------

            weights    = self.weights
            mid_vols   = self.mid_vols  
            expiries   = self.expiries
            calls      = model_output.fits[self.index,:,:]
            
            assert calls.shape == weights.shape, f"Expected calls and weights to have the same shape, but found {calls.shape} and {weights.shape}"
            assert np.all(np.isfinite(calls)), f"Expected all calls to be finite, but found {np.sum(~np.isfinite(calls))} non-finite values"

            model_vols = bs.implied( k=self.mkt_strikes, prices=calls, sqrtT=np.sqrt(self.expiries[:,None]), on_exceed_bounds="quiet" )
            diff_vols  = np.abs(model_vols - self.mid_vols)

            self.store.clear()
            self.store += self.axFittedVols.plot_surface(self.mkt_exp_block, self.mkt_fig_strikes, model_vols, cmap='viridis', edgecolor='none')
            self.store += self.axAbsVolsDiff.plot_surface(self.mkt_exp_block, self.mkt_fig_strikes, np.clip( diff_vols, 0., 0.05), cmap='viridis', edgecolor='none')
            self.store += self.axWeightedAbsVolsDiff.plot_surface(self.mkt_exp_block, self.mkt_fig_strikes, np.clip( weights*diff_vols, 0., 0.05), cmap='viridis', edgecolor='none')

            # model coordinates
            # -----------------

            sqrtVars    = model_output.sqrtVars[self.index,:]
            dlvsqs      = model_output.dlvsqs[self.index,:,:]
            xdensities  = model_output.xdensities[self.index,:,:]
            xstrikes    = model_output.xstrikes[self.index,:,:]
            model_viz_K = ( np.log(xstrikes) / sqrtVars[:,None] ) if self.viz_config.use_nms else xstrikes
            state       = model_output.state[self.index,:]
            assert dlvsqs.shape == xdensities[:,1:-1].shape, f"Expected dlvsqs and xdensities[:,1:-1] to have the same shape, but found {dlvsqs.shape} and {xdensities[:,1:-1].shape}"

            mdl_xpyblock = np.stack( [ self.expiries*DAYS_PER_YEAR ]*(xstrikes.shape[-1]), axis=-1 )
            model_dlvs   = np.sqrt( model_output.dlvsqs[self.index,:,:] )
            assert model_dlvs.shape == xstrikes[:,1:-1].shape, f"Expected model_dlvs and xstrikes[:,1:-1] to have the same shape, but found {model_dlvs.shape} and {xstrikes[:,1:-1].shape}"
            x = np.reshape(mdl_xpyblock[:,1:-1], (-1,))
            y = np.reshape(model_viz_K[:,1:-1], (-1,))
            dx = (x.max() - x.min())/100.
            dy = (y.max() - y.min())/100.
            z = np.reshape(model_dlvs,(-1,))
            assert x.shape == y.shape == z.shape, f"Expected x, y and z to have the same shape, but found {x.shape}, {y.shape} and {z.shape}"
            self.store += self.axDLVs.bar3d(x=x, y=y, z=z*0., dx=dx, dy=dy, dz=z, cmap="viridis", edgecolor='none' )

            model_density = model_output.xdensities[self.index,:,:]
            assert model_density.shape == xstrikes.shape, f"Expected model_density and xstrikes to have the same shape, but found {model_density.shape} and {xstrikes.shape}"
            x = np.reshape(mdl_xpyblock, (-1,))
            y = np.reshape(model_viz_K, (-1,))
            dx = (x.max() - x.min())/100.
            dy = (y.max() - y.min())/100.
            z = np.reshape(model_density,(-1,))
            self.store += self.axDensity.bar3d(x=x, y=y, z=z*0., dx=dx, dy=dy, dz=z, cmap="viridis", edgecolor='none' )

            self.store += self.axFwdVols.plot( np.sqrt(expiries), np.sqrt( np.diff(sqrtVars**2,prepend=0.)/np.diff(expiries, prepend=0.) ), color="orange", label="Model")
            if self.first:
                self.axFwdVols.legend(loc="upper left")

            self.store += self.axState.bar( np.arange(self.mdl_config.num_state),state, width=0.2, color="orange" )
            self.axState.set_title(f"State epoch {epoch}")
            self.fig.render() if not final else self.fig.close()

            self.first = False

    def __init__(self, market, *, 
                       model_output0 : PrettyObject,
                       viz_config : VizConfig, 
                       mdl_config : SanosModelConfig ):
        self.market        = market
        self.mdl_config    = mdl_config
        self.viz_config    = viz_config
        self.store         = FigStore()
        self.ticker        = self.market.grid_market_def.ticker
        self.first         = True

        # learning & fit
        # --------------

        self.fig_lrn   = figure(f"Loss {self.ticker}", col_size=8, row_size=4, columns=None) #
        
        self.ax_lrn    = self.fig_lrn.add_subplot("Loss")
        self.ax_lrn.set_xlabel("Epoch")
        self.ax_lrn.set_ylabel("Log-Loss")
        self.ax_lrn.set_yscale("log")

        self.ax_error = self.fig_lrn.add_subplot(r"Stats per date of wgt*(Fit-Mid)[wgt>0] ~ $\sigma\sqrt{t}$ error")
        self.ax_error.set_xlabel("Date")
        self.ax_error.set_ylabel("Error")
        self.ax_error.tick_params(axis="x", labelrotation=90)

        self.ax_herror = self.fig_lrn.add_subplot(r"Stats per date of Huber(wgt*(Fit-Mid)))[wgt>0] ~ $\sigma\sqrt{t}$ error")
        self.ax_herror.set_xlabel("Date")
        self.ax_herror.set_ylabel("Huber Error")
        self.ax_herror.tick_params(axis="x", labelrotation=90)

        self.fig_lrn.render()

        # states
        # ------

        states0        = model_output0.state
        assert states0.shape[1] == mdl_config.num_state, f"Expected states0.shape[1] {states0.shape[1]} to be equal to mdl_config.num_state {mdl_config.num_state}"
        self.ax_states = []
        self.fig_state = figure(f"State", col_size=4, row_size=4, columns=min(mdl_config.num_state,10) )
        colors         = colors_tableau()
        for istate in range(mdl_config.num_state):
            ax = self.fig_state.add_subplot(f"state {istate}")
            if istate>0:
                ax.sharey(self.ax_states[0])
            self.ax_states.append(ax)
            self.store += ax.plot(self.market.dates, states0[:,istate], color=next(colors))
            ax.set_xlabel("Date")
            ax.tick_params(axis="x", labelrotation=90)
        self.fig_state.render()
        del states0
        del colors

        # vols
        # ----

        np.random.seed(viz_config.seed)
        self.indices     = sorted( np.random.choice( np.arange(len(self.market.dates)), size=min(viz_config.num_viz_dates, len(self.market.dates)), replace=False ) ) \
                           if not isinstance(viz_config.num_viz_dates, list) else \
                            sorted( viz_config.num_viz_dates )
        self.indices     = self.indices[::-1] # reverse so that the earliest date is plotted first
        self.expiry_gfxs = [ self.ExpiryGfx(ticker=self.ticker,
                                            date =self.market.dates[ix], 
                                            index=ix,
                                            market=self.market, 
                                            model_output0=model_output0,
                                            mdl_config=mdl_config, 
                                            viz_config=viz_config ) for ix in self.indices ]
        assert len(self.expiry_gfxs) == len(self.indices), f"Expected len(expiry_gfxs) {len(self.expiry_gfxs)} to be equal to len(indices) {len(self.indices)}" 

    def __call__(self, tracker, num_total_epochs, *, model_output, error_stats, final : bool = False ):
        """
        Update
        """

        self.store.clear()

        # losses
        # ------
        trn_epochs =  np.asarray(tracker.trn_epochs)
        trn_losses =  np.asarray(tracker.trn_losses) 
        val_epochs =  np.asarray(tracker.val_epochs)
        val_losses =  np.asarray(tracker.val_losses) 
        epoch     =   trn_epochs[-1]
        self.fig_lrn.suptitle(f"Training loss {trn_losses[-1]:.4g} [{trn_epochs[-1]}], validation loss {val_losses[-1]:.4g} [{val_epochs[-1]}]. Total epochs {num_total_epochs}." + ("\n(Done)" if final else ""))

        self.store += self.ax_lrn.plot(trn_epochs, trn_losses, color="blue", label="Batch Loss")
        self.store += self.ax_lrn.plot(val_epochs, val_losses, "*:", color="orange", label="Full Loss")
        max_t = np.max(trn_losses) if len(trn_losses)<10 else np.quantile(trn_losses, 0.9)
        max_v = np.max(val_losses) if len(val_losses)<10 else np.quantile(val_losses, 0.9)
        max_  = min( max( max_t, max_v ), 1. )
        self.ax_lrn.set_ylim( max(0.,min( np.min(trn_losses), np.min(val_losses), )*0.9), max_ * 1.1 )
        if self.first:
            self.ax_lrn.legend(loc="lower left")

        medians = error_stats.raw_error_stats.medians
        means   = error_stats.raw_error_stats.means
        p90     = error_stats.raw_error_stats.p90
        assert medians.shape == means.shape == p90.shape == (len(self.market.dates),), f"Expected medians, means, and p90 to have shape {(len(self.market.dates),)}, but found {medians.shape}, {means.shape}, and {p90.shape}"
        self.store += self.ax_error.plot(self.market.dates, p90, "--", color="red", label="p90")
        self.store += self.ax_error.plot(self.market.dates, means, ".-", color="blue", label="Mean")
        self.store += self.ax_error.plot(self.market.dates, medians, ".-", color="orange", label="Median")
        if self.first:
            self.ax_error.legend(loc="upper left")
        self.ax_error.set_ylim(0., min( max( np.max(medians), np.max(means) )*1.1, 0.005 ) )

        medians = error_stats.error_stats.medians
        means   = error_stats.error_stats.means
        p90     = error_stats.error_stats.p90
        assert medians.shape == means.shape == p90.shape == (len(self.market.dates),), f"Expected medians, means, and p90 to have shape {(len(self.market.dates),)}, but found {medians.shape}, {means.shape}, and {p90.shape}"
        self.store += self.ax_herror.plot(self.market.dates, p90, "--", color="red", label="p90")
        self.store += self.ax_herror.plot(self.market.dates, means, ".-", color="blue", label="Mean")
        self.store += self.ax_herror.plot(self.market.dates, medians, ".-", color="orange", label="Median")
        if self.first:
            self.ax_herror.legend(loc="upper left")
        self.ax_herror.set_ylim(0.,max( np.max(medians), np.max(means) )*1.1 )

        self.fig_lrn.render() if not final else self.fig_lrn.close()

        # states
        # ------

        assert model_output.state.shape[1] == len(self.ax_states), f"Expected state to have shape[1] == {len(self.ax_states)}, but found {model_output.state.shape[1]}"   
        colors  = colors_tableau()
        for istate, axstate in enumerate(self.ax_states):
            self.store += axstate.plot(self.market.dates, model_output.state[:,istate], color=next(colors))
        self.ax_states[0].set_ylim( m_o_m( model_output.state ) )
        self.fig_state.suptitle(f"States at epoch {epoch} / {num_total_epochs}")
        self.fig_state.render() if not final else self.fig_state.close()

        # rest
        for gfx in self.expiry_gfxs:
            assert gfx.index<model_output.fits.shape[0], f"Expected ix {gfx.index} to be less than fits.shape[0] {model_output.fits.shape[0]}"
            gfx.update( epoch=epoch, model_output=model_output, final=final )

        self.first = False
