import numpy as np
from cdxcore.dynaplot import FigStore, colors_tableau, figure, m_o_m, quantile_m_o_m
from cdxcore.bs import bs

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_model import SanosCompressorConfig

class Display(object):

    class ExpiryGfx(object):

        def __init__(self, ticker, *, date, index, ts, mdl_config : SanosCompressorConfig, trn_config, isNMS ):
            self.fig        = figure(f"{ticker} {date}", col_size=5, row_size=5, columns=None )
            self.index      = index
            self.date       = date
            self.ts         = ts
            self.mdl_config = mdl_config
            self.trn_config = trn_config
            self.store      = FigStore()
            self.isNMS      = isNMS

            self.expiries = ts['expiries']
            assert np.max(self.index) < self.expiries.shape[0], f"Expected index {self.index} to be less than expiries.shape[0] {self.expiries.shape[0]}"

            self.expiries = self.expiries[self.index,:]
            self.xstrikes = ts['xstrikes'][self.index,:,:]
            self.nms      = ts['nms'][self.index,:,:]
            self.mids     = ts['mids'][self.index,:,:]
            self.mid_vols = ts['mid_vols'][self.index,:,:]

            fit_vols      = ts['mid_vols'][self.index,:,:]
            fit_dlvsqs    = ts['fit_dlvsqs'][self.index,:,:]

            strikes       = self.nms if self.isNMS else self.xstrikes[:,1:-1]
            strk_name     = "NMS" if self.isNMS else "K"
    
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
            strikes    = self.nms if self.isNMS else self.xstrikes[:,1:-1]
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

    def __init__(self, ticker, *, data, num_viz_dates, seed_viz_dates, isNMS, mdl_config, trn_config ):
        self.data          = data
        self.num_viz_dates = num_viz_dates
        self.isNMS         = isNMS
        self.mdl_config    = mdl_config
        self.trn_config    = trn_config
        self.store         = FigStore()
        np.random.seed(seed_viz_dates)

        self._figs = {}
        zero_dates     = np.zeros((len(self.data.ts.dates),))

        # learning
        self.fig_lrn = figure(f"Learning rate {ticker}", col_size=8, row_size=4, columns=None) 
        self.ax_lrn  = self.fig_lrn.add_subplot("Loss")
        self.ax_lrn.set_xlabel("Epoch")
        self.ax_lrn.set_ylabel("Log-Loss")
        self.ax_lrn.set_yscale("log")

        self.ax_error = self.fig_lrn.add_subplot("Mean and Median of err[wgt>0] divided by mean(wgt[wgt>0])")
        self.store += self.ax_error.plot(self.data.ts.dates, zero_dates, ".-", color="orange", label="Median")
        self.store += self.ax_error.plot(self.data.ts.dates, zero_dates, ".-", color="blue", label="Mean")
        self.ax_error.set_xlabel("Date")
        self.ax_error.set_ylabel("Error")
        self.ax_error.tick_params(axis="x", labelrotation=90)
        self.ax_error.legend(loc="upper left")

        self.fig_lrn.render()

        # state
        dates          = data.ts.dates
        self.ax_states = []
        self.fig_state = figure(f"State", col_size=4, row_size=4, columns=min(mdl_config.num_state,10) )
        colors         = colors_tableau()
        self.ax_states = []
        for istate in range(mdl_config.num_state):
            ax = self.fig_state.add_subplot(f"state {istate}")
            if istate>0:
                ax.sharey(self.ax_states[0])
            self.ax_states.append(ax)
            self.store += ax.plot(self.data.ts.dates, zero_dates, color=next(colors))
            ax.set_xlabel("Date")
            ax.tick_params(axis="x", labelrotation=90)
        self.fig_state.render()

        np.random.seed(123123)
        self.indices     = sorted( np.random.choice( np.arange(len(dates)), size=min(self.num_viz_dates, len(dates)), replace=False ) )
        self.expiry_gfxs = [ self.ExpiryGfx(ticker=ticker, date =dates[ix], index=ix, ts=self.data.ts, mdl_config=mdl_config, trn_config=trn_config, isNMS=self.isNMS) for ix in self.indices ]
        assert len(self.expiry_gfxs) == len(self.indices), f"Expected len(expiry_gfxs) {len(self.expiry_gfxs)} to be equal to len(indices) {len(self.indices)}" 

    def __call__(self, tracker, *, calls, xdensities, dlvsqs, errors, weights, state, final : bool = False ):

        self.store.clear()

        # losses
        trn_epochs =  np.asarray(tracker.trn_epochs)
        trn_losses =  np.asarray(tracker.trn_losses) 
        val_epochs =  np.asarray(tracker.val_epochs)
        val_losses =  np.asarray(tracker.val_losses) 
        epoch     =   trn_epochs[-1]
        self.store += self.ax_lrn.plot(val_epochs, val_losses, "*:", color="orange", label="Full Loss")
        self.store += self.ax_lrn.plot(trn_epochs, trn_losses, color="blue", label="Batch Loss")
        max_t = np.max(trn_losses) if len(trn_losses)<10 else np.quantile(trn_losses, 0.9)
        max_v = np.max(val_losses) if len(val_losses)<10 else np.quantile(val_losses, 0.9)
        max_  = max( max_t, max_v )
        self.ax_lrn.set_ylim( min( np.min(trn_losses), np.min(val_losses) )*0.9, max_ * 1.1 )
        self.ax_lrn.legend(loc="lower left")
        self.fig_lrn.suptitle(f"Training loss {trn_losses[-1]:.4g} [{trn_epochs[-1]}], validation loss {val_losses[-1]:.4g} [{val_epochs[-1]}]. Total epochs {self.trn_config.num_epochs}.")

        errors  = [ errors[i,...][weights[i,...]>0] for i in range(errors.shape[0]) ]
        weights = [ np.mean( weights[i,...][weights[i,...]>0] ) for i in range(weights.shape[0]) ]
        medians = [ np.median(err)/w for err, w in zip(errors, weights) ]
        means   = [ np.mean(err)/w for err, w in zip(errors, weights) ]
        medians = np.array(medians)
        means   = np.array(means)
        assert medians.shape == means.shape == (len(self.data.ts.dates),), f"Expected medians and means to have shape {(len(self.data.ts.dates),)}, but found {medians.shape} and {means.shape}"
        self.store += self.ax_error.plot(self.data.ts.dates, medians, ".-", color="orange")
        self.store += self.ax_error.plot(self.data.ts.dates, means, ".-", color="blue")
        self.ax_error.set_ylim(0.,max( np.max(medians), np.max(means) )*1.1 )
        self.fig_lrn.render() if not final else self.fig_lrn.close()

        # states
        assert state.shape[1] == len(self.ax_states), f"Expected state to have shape[1] == {len(self.ax_states)}, but found {state.shape[1]}"   
        colors  = colors_tableau()
        for istate, axstate in enumerate(self.ax_states):
            self.store += axstate.plot(self.data.ts.dates, state[:,istate], color=next(colors))
        self.ax_states[0].set_ylim( m_o_m( state ) )
        self.fig_state.suptitle(f"State at epoch {epoch} / {self.trn_config.num_epochs}")
        self.fig_state.render() if not final else self.fig_state.close()

        # rest
        for gfx in self.expiry_gfxs:
            assert gfx.index<calls.shape[0], f"Expected ix {gfx.index} to be less than calls.shape[0] {calls.shape[0]}"
            gfx.update( epoch=epoch, 
                        calls=calls,
                        xdensities=xdensities,
                        dlvsqs=dlvsqs, 
                        state=state, 
                        final=final )
