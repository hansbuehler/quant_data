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
    num_viz_dates : int = 4       # how many implied vols to render
    seed          : int = 1231231 # seed for chosing the above
    use_nms       : bool = True   # whether use NMS coordinates or K

    def __post_init__(self):
        assert self.num_viz_dates > 0, f"Expected 'num_viz_dates' {self.num_viz_dates} to be greater than 0"
        assert self.seed >= 0, f"Expected 'seed' {self.seed} to be greater than or equal to 0"

class Visualizer(object):

    class ExpiryGfx(object):

        def __init__(self, ticker, *, 
                           date, index, ts, 
                           viz_config : VizConfig,
                           mdl_config : SanosModelConfig ):
            self.fig        = figure(f"{ticker} {date}", col_size=5, row_size=5, columns=None )
            self.index      = index
            self.date       = date
            self.ts         = ts
            self.viz_config = viz_config    
            self.mdl_config = mdl_config
            self.store      = FigStore()

            self.expiries = ts['expiries']
            assert np.max(self.index) < self.expiries.shape[0], f"Expected index {self.index} to be less than expiries.shape[0] {self.expiries.shape[0]}"

            self.expiries = self.expiries[self.index,:]
            self.xstrikes = ts['xstrikes'][self.index,:,:]
            self.nms      = ts['nms'][self.index,:,:]
            self.mids     = ts['mids'][self.index,:,:]
            self.mid_vols = ts['mid_vols'][self.index,:,:]

            fit_vols      = ts['mid_vols'][self.index,:,:]
            fit_dlvsqs    = ts['fit_dlvsqs'][self.index,:,:]

            strikes       = self.nms if self.viz_config.use_nms else self.xstrikes[:,1:-1]
            strk_name     = "NMS" if self.viz_config.use_nms else "K"
    
            self.expiries_block = np.stack( [ self.expiries*DAYS_PER_YEAR ]*(self.xstrikes.shape[-1]-2), axis=-1 )

            assert self.expiries_block.shape == self.nms.shape == self.mid_vols.shape, f"Expected expiries_block, nms and mid_vols to have the same shape, but found {self.expiries_block.shape}, {self.nms.shape} and {self.mid_vols.shape}"
            self.axMarketVols = self.fig.add_subplot("Interpolated Market Vols", projection="3d") 
            self.axMarketVols.plot_surface( self.expiries_block, strikes, self.mid_vols, cmap='viridis', edgecolor='none')
            self.axMarketVols.set_xlabel("Days")
            self.axMarketVols.set_ylabel(strk_name)
            self.axMarketVols.view_init(elev=25, azim=-45) 

            assert self.expiries_block.shape == self.nms.shape == fit_vols.shape, f"Expected expiries_block, nms and fit_vols to have the same shape, but found {self.expiries_block.shape}, {self.nms.shape} and {fit_vols.shape}"
            self.axFittedVols = self.fig.add_subplot("Model Vols", projection="3d")
            self.axFittedVols.set_xlabel("Days")
            self.axFittedVols.set_ylabel(strk_name)
            self.axFittedVols.view_init(elev=25, azim=-45) 

            assert self.expiries_block.shape == self.nms.shape == np.abs(fit_vols - self.mid_vols).shape, f"Expected expiries_block, nms and fit_vols to have the same shape, but found {self.expiries_block.shape}, {self.nms.shape} and {np.abs(fit_vols - self.mid_vols).shape}"
            self.axAbsVolsDiff = self.fig.add_subplot("Abs Vols Difference", projection="3d")
            self.axAbsVolsDiff.set_xlabel("Days")
            self.axAbsVolsDiff.set_ylabel(strk_name)
            self.axAbsVolsDiff.set_zlim( 0., 0.05 )
            self.axAbsVolsDiff.view_init(elev=25, azim=-45) 

            x = np.reshape(self.expiries_block, (-1,))
            y = np.reshape(strikes, (-1,))
            dx = (x.max() - x.min())/100.
            dy = (y.max() - y.min())/100.

            z = np.reshape(np.sqrt(fit_dlvsqs),(-1,))
            assert x.shape == y.shape == z.shape, f"Expected x, y and z to have the same shape, but found {x.shape}, {y.shape} and {z.shape}"
            self.axDLVs = self.fig.add_subplot("DLVs", projection="3d")
            self.store += self.axDLVs.bar3d(x=x, y=y, z=0.*z, dx=dx, dy=dy, dz=z, cmap="viridis", edgecolor='none' )
            self.axDLVs.set_xlabel("Days")
            self.axDLVs.set_ylabel(strk_name)
            self.axDLVs.set_zlabel("DLV")
            self.axDLVs.view_init(elev=25, azim=-45) 

            self.axDensity = self.fig.add_subplot("Density", projection="3d")
            #z = np.reshape(np.sqrt(fit_dlvsqs),(-1,))
            #self.store += self.axDensity.bar3d(x=x, y=y, z=0.*z, dx=dx, dy=dy, dz=z, cmap="viridis", edgecolor='none' )
            self.axDensity.set_xlabel("Days")
            self.axDensity.set_ylabel(strk_name)
            self.axDensity.set_zlabel("Density")
            self.axDensity.view_init(elev=25, azim=-45)

            # state
            self.axState = self.fig.add_subplot("State")            
            self.store += self.axState.bar( np.arange(mdl_config.num_state),np.zeros((mdl_config.num_state,)), width=0.2 )

            self.fig.render()

        def update(self, *, epoch, calls, dlvsqs, xdensities, state, final : bool = False ):
            """ update this date """
            #print(f"/n{epoch} {self.index} {self.date}")
            calls      = calls[self.index,:,:]
            xdensities = xdensities[self.index,:,:]
            dlvsqs     = dlvsqs[self.index,:,:]
            state      = state[self.index,:]
            strikes    = self.nms if self.viz_config.use_nms else self.xstrikes[:,1:-1]
            x = np.reshape(self.expiries_block, (-1,))
            y = np.reshape(strikes, (-1,))
            dx = (x.max() - x.min())/100.
            dy = (y.max() - y.min())/100.

            assert np.all(np.isfinite(calls)), f"Expected all calls to be finite, but found {np.sum(~np.isfinite(calls))} non-finite values"
            vols = bs.implied( k=self.xstrikes[:,1:-1], prices=calls, sqrtT=np.sqrt(self.expiries[:,None]) )

            self.store.clear()
            self.store += self.axFittedVols.plot_surface(self.expiries_block, strikes, vols, cmap='viridis', edgecolor='none')
            self.store += self.axAbsVolsDiff.plot_surface(self.expiries_block, strikes, np.clip( np.abs(vols - self.mid_vols), 0., 0.05), cmap='viridis', edgecolor='none')

            z = np.reshape(np.sqrt(dlvsqs),(-1,))
            self.store += self.axDLVs.bar3d(x=x, y=y, z=0.*z, dx=dx, dy=dy, dz=z, cmap="viridis", edgecolor='none' )

            z = np.reshape(xdensities[:,1:-1],(-1,))
            self.store += self.axDensity.bar3d(x=x, y=y, z=0.*z, dx=dx, dy=dy, dz=z, cmap="viridis", edgecolor='none' )

            self.store += self.axState.bar( np.arange(self.mdl_config.num_state),state, width=0.2, color="orange" )
            self.axState.set_title(f"State epoch {epoch}")
            self.fig.render() if not final else self.fig.close()

    def __init__(self, market, *, 
                       viz_config : VizConfig, 
                       mdl_config : SanosModelConfig ):
        self.market        = market
        self.mdl_config    = mdl_config
        self.viz_config    = viz_config
        self.store         = FigStore()
        self.ticker        = self.market.grid_market_def.ticker
        self.first         = True
        np.random.seed(self.viz_config.seed)

        all_dates          = self.market.dates
        zero_ndates        = np.zeros((len(all_dates),))

        # learning & fit
        # --------------

        self.fig_lrn   = figure(f"Learning rate {self.ticker}", col_size=8, row_size=4, columns=None) #
        
        self.ax_lrn    = self.fig_lrn.add_subplot("Loss")
        self.ax_lrn.set_xlabel("Epoch")
        self.ax_lrn.set_ylabel("Log-Loss")
        self.ax_lrn.set_yscale("log")

        self.ax_error = self.fig_lrn.add_subplot(r"Stats per date of err[wgt>0] ~ $\sigma\sqrt{t}$ error")
        self.ax_error.set_xlabel("Date")
        self.ax_error.set_ylabel("Error")
        self.ax_error.tick_params(axis="x", labelrotation=90)

        self.fig_lrn.render()

        # states
        # ------

        self.ax_states = []
        self.fig_state = figure(f"State", col_size=4, row_size=4, columns=min(mdl_config.num_state,10) )
        colors         = colors_tableau()
        self.ax_states = []
        for istate in range(mdl_config.num_state):
            ax = self.fig_state.add_subplot(f"state {istate}")
            if istate>0:
                ax.sharey(self.ax_states[0])
            self.ax_states.append(ax)
            ax.set_xlabel("Date")
            ax.tick_params(axis="x", labelrotation=90)
        self.fig_state.render()

        # vols
        # ----

        np.random.seed(123123)
        self.indices     = sorted( np.random.choice( np.arange(len(all_dates)), size=min(viz_config.num_viz_dates, len(all_dates)), replace=False ) )
        self.indices     = self.indices[::-1] # reverse so that the earliest date is plotted first
        self.expiry_gfxs = [ self.ExpiryGfx(ticker=self.ticker, date =all_dates[ix], index=ix, ts=self.market, mdl_config=mdl_config, viz_config=viz_config ) for ix in self.indices ]
        assert len(self.expiry_gfxs) == len(self.indices), f"Expected len(expiry_gfxs) {len(self.expiry_gfxs)} to be equal to len(indices) {len(self.indices)}" 

    def __call__(self, tracker, num_total_epochs, *, data, final : bool = False ):
        """
        Update
        """

        self.store.clear()

        # losses
        trn_epochs =  np.asarray(tracker.trn_epochs)
        trn_losses =  np.asarray(tracker.trn_losses) 
        val_epochs =  np.asarray(tracker.val_epochs)
        val_losses =  np.asarray(tracker.val_losses) 
        epoch     =   trn_epochs[-1]
        self.fig_lrn.suptitle(f"Training loss {trn_losses[-1]:.4g} [{trn_epochs[-1]}], validation loss {val_losses[-1]:.4g} [{val_epochs[-1]}]. Total epochs {num_total_epochs}.")

        self.store += self.ax_lrn.plot(trn_epochs, trn_losses, color="blue", label="Batch Loss")
        self.store += self.ax_lrn.plot(val_epochs, val_losses, "*:", color="orange", label="Full Loss")
        max_t = np.max(trn_losses) if len(trn_losses)<10 else np.quantile(trn_losses, 0.9)
        max_v = np.max(val_losses) if len(val_losses)<10 else np.quantile(val_losses, 0.9)
        max_  = max( max_t, max_v )
        self.ax_lrn.set_ylim( min( np.min(trn_losses), np.min(val_losses) )*0.9, max_ * 1.1 )
        if self.first:
            self.ax_lrn.legend(loc="lower left")

        medians = data.error_stats.medians
        means   = data.error_stats.means
        p90     = data.error_stats.p90
        assert medians.shape == means.shape == p90.shape == (len(self.market.dates),), f"Expected medians, means, and p90 to have shape {(len(self.market.dates),)}, but found {medians.shape}, {means.shape}, and {p90.shape}"
        self.store += self.ax_error.plot(self.market.dates, p90, "--", color="red", label="p90")
        self.store += self.ax_error.plot(self.market.dates, means, ".-", color="blue", label="Mean")
        self.store += self.ax_error.plot(self.market.dates, medians, ".-", color="orange", label="Median")
        if self.first:
            self.ax_error.legend(loc="upper left")

        self.ax_error.set_ylim(0.,max( np.max(medians), np.max(means) )*1.1 )
        self.fig_lrn.render() if not final else self.fig_lrn.close()

        # states
        assert data.state.shape[1] == len(self.ax_states), f"Expected state to have shape[1] == {len(self.ax_states)}, but found {data.state.shape[1]}"   
        colors  = colors_tableau()
        for istate, axstate in enumerate(self.ax_states):
            self.store += axstate.plot(self.market.dates, data.state[:,istate], color=next(colors))
        self.ax_states[0].set_ylim( m_o_m( data.state ) )
        self.fig_state.suptitle(f"State at epoch {epoch} / {num_total_epochs}")
        self.fig_state.render() if not final else self.fig_state.close()

        # rest
        for gfx in self.expiry_gfxs:
            assert gfx.index<data.calls.shape[0], f"Expected ix {gfx.index} to be less than calls.shape[0] {data.calls.shape[0]}"
            gfx.update( epoch=epoch, 
                        calls=data.calls,
                        xdensities=data.xdensities,
                        dlvsqs=data.dlvsqs, 
                        state=data.state, 
                        final=final )

        self.first = False
