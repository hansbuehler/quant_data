"""Tests for dys.calendar."""

from __future__ import annotations

import numpy as np
import math as math
import pytest

from grid_sanos_data_test import TEST_DATA
from quantdata.sanos.grid import GridGeometry, GridSanos, Eps, BackboneNumpy
from quantdata.sanos.grid.fitter import Fitter
from quantdata.torch.bbtorch import BackboneTorch
from cdxcore.pretty import PrettyObject
from cdxcore.verbose import Context
from cdxcore.bs import bs

class TestGridSanos:

    @staticmethod
    def _make_grid(vol_interp: str = "lin_sqrt_time") -> GridGeometry:
        return GridGeometry.create_grid_nms(
            horizon_t=1.,
            num_times=8,
            nms_range=(-4,4),
            num_nms=18,
            dlv_min=0.01,
            dlv_max=5.,
            fwdvol_min=0.01,
            vol_interp=vol_interp,
            lower_scale=0.2,
            upper_scale=1.5,
            dtype=np.float64,
        )

    @staticmethod
    def _market_examples(num_examples: int = 7) -> PrettyObject:
        unique_tte = np.unique(TEST_DATA["tte_y"].astype(np.float64))
        assert len(unique_tte) >= num_examples, f"Expected at least {num_examples} unique expiries, found {len(unique_tte)}"

        positions = np.linspace(0, len(unique_tte) - 1, num_examples, dtype=int)
        sample_tte = unique_tte[positions]
        sample_ix = [int(np.flatnonzero(TEST_DATA["tte_y"] == tte)[0]) for tte in sample_tte]

        opt_mask = np.ones((len(sample_ix),), dtype=bool)
        if len(opt_mask) > 2:
            opt_mask[1] = False
            opt_mask[-2] = False

        opt_is_call = np.array([i % 2 == 0 for i in range(len(sample_ix))], dtype=bool)
        return PrettyObject(
            strikes=np.asarray(TEST_DATA["strikes"][sample_ix], dtype=np.float64),
            tte_y=np.asarray(TEST_DATA["tte_y"][sample_ix], dtype=np.float64),
            opt_mask=opt_mask,
            opt_is_call=opt_is_call,
        )

    @staticmethod
    def _surface_examples(grid: GridGeometry) -> np.ndarray:
        T, N = grid.TN
        time_axis = np.linspace(0.0, 1.0, T, dtype=np.float64)[:, None]
        strike_axis = np.linspace(0.0, 1.0, N - 2, dtype=np.float64)[None, :]

        dlvsqs = np.stack([
            np.zeros((T, N - 2), dtype=np.float64),
            0.01 + 0.04 * (0.5 * time_axis + strike_axis),
            0.02 + 0.03 * np.sin(0.5 * np.pi * time_axis) * (1.0 + strike_axis),
        ], axis=0)
        assert dlvsqs.shape == (3, T, N - 2), f"Expected example surfaces to have shape {(3, T, N - 2)}, found {dlvsqs.shape}"
        return dlvsqs

    @staticmethod
    def _assert_numpy_torch_close(np_value, tr_value, *, atol: float = 1e-9, label: str = "values"):
        tr_value = tr_value.cpu().numpy()
        assert np_value.shape == tr_value.shape, f"Expected matching shapes for {label}, found {np_value.shape} and {tr_value.shape}"
        assert np.allclose(np_value, tr_value, atol=atol), f"Expected numpy and torch {label} to match, max abs diff {np.max(np.abs(np_value - tr_value))}"

    def test_roundtrip(self):

        # fill round trip test

        def run_data( r, FWD_VOLS  ):
            assert isinstance(r, dict)
            r           = PrettyObject(r)
            grid        = GridGeometry.create_grid_nms(
                                    horizon_t  = 1., 
                                    num_times  = 8,
                                    nms_range  = (-4,4),
                                    num_nms    = 30,
                                    dlv_min    = 0.01,
                                    dlv_max    = 5.,
                                    fwdvol_min = 0.0,
                                    vol_interp = "lin_var",
                                    lower_scale = 0.2,
                                    upper_scale = 1.5,
                                    dtype = np.float64
                )

            eps         = Eps()
            fitter      = Fitter( geometry = grid, eps = eps, debug=True, max_error = None, mid_fit_weight = 0.1 )
            sanos       = GridSanos( geometry = grid, bb = BackboneNumpy() )

            # fit to market data
            dlvsqs, fit_prices, densities = \
                fitter( fwd_vols     = FWD_VOLS,
                        opt_strikes  = r.strikes,
                        opt_bid      = r.bid,
                        opt_ask      = r.ask,
                        opt_tte      = r.tte_y,
                        opt_is_calls = None,
                        opt_mask     = None,
                        return_only_dlvsqs = False,
                        verbose      = Context.all ) 
            assert fit_prices.shape == r.strikes.shape, f"Expected fit_prices to have shape {r.strikes.shape}, got {fit_prices.shape}"
            fit_vols  = bs.implied(r.strikes, fit_prices, sqrtT = np.sqrt(r.tte_y) )

            # reprice with fitted density -> price
            re_prices = sanos.price(  fwd_vols=FWD_VOLS, 
                                      densities=densities,
                                      opt_strikes  = r.strikes,
                                      opt_tte      = r.tte_y
                                )
            re_vols  = bs.implied(r.strikes, re_prices, sqrtT = np.sqrt(r.tte_y) )
            assert np.max( np.abs( re_vols - fit_vols ) ) < 1e-4, f"Repriced vols should match fit vols, max abs diff {np.max( np.abs( re_vols - fit_vols ) )}"

            # rerprice with fitted dlvsq -> densitity -> price
            test_dens_from_dlv  = sanos.densities( dlvsqs )
            dlv_prices = sanos.price( fwd_vols=FWD_VOLS, 
                                    densities=test_dens_from_dlv,
                                    opt_strikes  = r.strikes,
                                    opt_tte      = r.tte_y
                                )
            dlv_vols  = bs.implied(r.strikes, dlv_prices, sqrtT = np.sqrt(r.tte_y) )
            assert np.max( np.abs( dlv_vols - fit_vols ) ) < 0.01, f"DLV vols should match fit vols, max abs diff {np.max( np.abs( dlv_vols - fit_vols ) )}"

            # DLVs, manual
            # ------------

            xstrikes = np.asarray( grid.xstrikes, dtype=np.float64 )
            gstrikes = xstrikes[1:,1:-1]
            density0 = np.asarray( grid.density0, dtype=np.float64 )

            C_t   = np.maximum( gstrikes[:,None,:] - gstrikes[:,:,None], 0. ) * densities[:,None,:]
            C_t   = np.sum(C_t, axis=-1)
            C_tm1 = np.maximum( gstrikes[:-1,None,:]- gstrikes[1:,:,None], 0. )* densities[:-1,None,:]
            C_tm1 = np.sum(C_tm1, axis=-1)
            C_0m1 = np.maximum( xstrikes[0,1:-1][None,:] - gstrikes[0,:,None], 0. ) * density0[None,:]
            C_0m1 = np.sum(C_0m1, axis=-1)

            dC_t  = C_t[1:,:] - C_tm1
            dC_0  = C_t[0,:] - C_0m1

            # DLV formula us
            #
            # C(t,k^i) - \bar C(t,k^i) = 1/2 DLV^2 dt k^i^2 \partial^2 C / \partial k^2
            #
            # \partial^2 C / \partial k^2 = { dC_t - dC_tm1 }/{ 1/2 (K_{i+1} - K_{i-1}) } = p_t / { 1/2 (K_{i+1} - K_{i-1}) } 
            #
            # hence
            #
            # C(t,k^i) - \bar C(t-1,k^i) = DLV^2 dt {  k^i^2 / (K_{i+1} - K_{i-1}) } p_t

            dlvsq_max   = grid.dlv_max**2.
            dlvqs_min   = grid.dlv_min**2.

            dts         = np.diff( grid.xexpiries.astype(np.float64) )
            box         = gstrikes[:,1:-1]**2 / ( gstrikes[:,2:] - gstrikes[:,:-2] ) * dts[:,None]
            boxed       = box * densities[:,1:-1]
            dC           = np.zeros_like( boxed )
            dC[1:,:]    = dC_t[:,1:-1]
            dC[0,:]     = dC_0[1:-1]

            dC_zero     = dC <= fitter.eps.dlvsq
            too_small   = dC <= dlvqs_min * boxed
            too_large   = dC >= dlvsq_max * boxed

            divisor      = np.where( dC_zero|too_small|too_large, 1., boxed )
            dlvsqs2      = np.select( [ dC_zero, too_small, too_large],
                                        [ dlvqs_min, dlvqs_min, dlvsq_max ],
                                        default = dC / divisor )

            dlvsqs2      = np.where( np.isfinite(dlvsqs2), dlvsqs2, dlvqs_min)
            dlvsqs2      = np.clip( dlvsqs2,  dlvqs_min, dlvsq_max ) - dlvqs_min

            inner  = ( dlvsqs > 0. ) & ( dlvsqs < dlvsq_max-dlvqs_min )
            inner2 = ( dlvsqs2 > 0. ) & ( dlvsqs2 < dlvsq_max-dlvqs_min )
            assert np.all( inner == inner2 ), f"DLV^2 from DLV formula should be out of bounds in the same places as fitted DLV^2, expected {np.sum(inner)}/{inner.size} inner points, found {np.sum(inner2)}/{inner2.size} inner points"
            assert np.max( np.abs( dlvsqs2 - dlvsqs ) ) < 1e-5, f"DLV^2 from DLV formula should match fitted DLV^2, max abs diff {np.max( np.abs( dlvsqs2 - dlvsqs ) ) }"

            test_dens_from_manual_dlv = sanos.densities( dlvsqs2 )
            prices_man_dlv  = sanos.price(  
                                      fwd_vols=FWD_VOLS,
                                      densities=test_dens_from_manual_dlv,
                                      opt_strikes  = r.strikes,
                                      opt_tte      = r.tte_y
                                )
            assert np.max( np.abs( prices_man_dlv - dlv_prices ) ) < 1e-8, f"Manual DLV prices should match prices from fitted DLV^2, max abs diff {np.max( np.abs( prices_man_dlv - dlv_prices ) )}"

        run_data( TEST_DATA, FWD_VOLS=0.0 ) # linear
        run_data( TEST_DATA, FWD_VOLS=0.05 )

    @pytest.mark.parametrize("vol_interp", ["lin_var", "lin_sqrt_time"])
    def test_numpy_and_torch_interpolate_sqrtvars_match_for_market_examples(self, vol_interp: str):

        grid = self._make_grid(vol_interp)
        market = self._market_examples()
        bbtr = BackboneTorch("cpu")
        np_sanos = GridSanos(grid, bb=BackboneNumpy())
        tr_sanos = GridSanos(grid, bb=bbtr)

        np_opt_interp = grid.time_interpolation_weights(market.tte_y, opt_mask=market.opt_mask)
        tr_opt_interp = {k: bbtr.to_array(v) for k, v in np_opt_interp.items()}
        tr_mask = bbtr.to_array(market.opt_mask)

        fwd_vol_examples = {
            "scalar": 0.03,
            "curve": np.linspace(0.0, 0.06, grid.T, dtype=np.float64),
        }

        for label, fwd_vols in fwd_vol_examples.items():
            np_sqrtvars = np_sanos.interpolate_sqrtVars(fwd_vols, opt_interp=np_opt_interp, opt_mask=market.opt_mask)
            tr_fwd_vols = fwd_vols if isinstance(fwd_vols, float) else bbtr.to_array(fwd_vols)
            tr_sqrtvars = tr_sanos.interpolate_sqrtVars(tr_fwd_vols, opt_interp=tr_opt_interp, opt_mask=tr_mask)

            self._assert_numpy_torch_close(np_sqrtvars, tr_sqrtvars, atol=1e-10, label=f"interpolate_sqrtVars[{vol_interp},{label}]")
            assert np.all(np_sqrtvars[~market.opt_mask] == 0.), f"Expected masked sqrt-variances to be zero for {vol_interp}/{label}"

    def test_numpy_and_torch_densities_match_for_example_surfaces(self):

        grid = self._make_grid()
        bbtr = BackboneTorch("cpu")
        np_sanos = GridSanos(grid, bb=BackboneNumpy())
        tr_sanos = GridSanos(grid, bb=bbtr)

        dlvsqs = self._surface_examples(grid)
        np_densities = np_sanos.densities(dlvsqs)
        tr_densities = tr_sanos.densities(bbtr.to_array(dlvsqs))

        self._assert_numpy_torch_close(np_densities, tr_densities, atol=1e-10, label="densities")
        assert np.allclose(np.sum(np_densities, axis=-1), 1.0, atol=1e-8), "Expected example densities to preserve total mass"

    @pytest.mark.parametrize("vol_interp", ["lin_var", "lin_sqrt_time"])
    def test_numpy_and_torch_price_match_for_market_examples(self, vol_interp: str):

        grid = self._make_grid(vol_interp)
        market = self._market_examples()
        bbtr = BackboneTorch("cpu")
        np_sanos = GridSanos(grid, bb=BackboneNumpy())
        tr_sanos = GridSanos(grid, bb=bbtr)

        np_opt_interp = grid.time_interpolation_weights(market.tte_y, opt_mask=market.opt_mask)
        tr_opt_interp = {k: bbtr.to_array(v) for k, v in np_opt_interp.items()}
        tr_strikes = bbtr.to_array(market.strikes)
        tr_tte = bbtr.to_array(market.tte_y)
        tr_mask = bbtr.to_array(market.opt_mask)
        tr_is_call = bbtr.to_array(market.opt_is_call)

        densities = np_sanos.densities(self._surface_examples(grid)[1])
        tr_densities = bbtr.to_array(densities)

        fwd_vol_examples = {
            "intrinsic": None,
            "scalar": 0.03,
            "curve": np.linspace(0.0, 0.05, grid.T, dtype=np.float64),
        }

        for label, fwd_vols in fwd_vol_examples.items():
            np_prices = np_sanos.price(
                fwd_vols=fwd_vols,
                densities=densities,
                opt_strikes=market.strikes,
                opt_tte=market.tte_y,
                opt_mask=market.opt_mask,
                opt_is_call=market.opt_is_call,
                opt_interp=np_opt_interp,
            )
            tr_fwd_vols = None if fwd_vols is None else (fwd_vols if isinstance(fwd_vols, float) else bbtr.to_array(fwd_vols))
            tr_prices = tr_sanos.price(
                fwd_vols=tr_fwd_vols,
                densities=tr_densities,
                opt_strikes=tr_strikes,
                opt_tte=tr_tte,
                opt_mask=tr_mask,
                opt_is_call=tr_is_call,
                opt_interp=tr_opt_interp,
            )

            self._assert_numpy_torch_close(np_prices, tr_prices, atol=1e-10, label=f"price[{vol_interp},{label}]")
            assert np.all(np_prices[~market.opt_mask] == 0.), f"Expected masked prices to be zero for {vol_interp}/{label}"

    def test_torch_samples(self):

            
        grid        = GridGeometry.create_grid_nms(
                                horizon_t = 1., 
                                num_times = 20,
                                nms_range = (-4,4),
                                num_nms   = 40,
                                dlv_min   = 0.01,
                                dlv_max   = 5.,
                                fwdvol_min       = 0.01,
                                vol_interp = "lin_sqrt_time",
                                lower_scale = 0.2,
                                upper_scale = 1.5,
                                dtype = np.float64
                )

        eps         = Eps()
        eps.set_all( 1E-3 )
        fitter      = Fitter( geometry = grid, eps = eps, debug=True, max_error = None, mid_fit_weight = 1. )
        bbnp        = BackboneNumpy()
        bbtr        = BackboneTorch("cpu")
        np_sanos    = GridSanos( grid, bb = bbnp )
        tr_sanos    = GridSanos( grid, bb = bbtr )

        # sample
        # -----

        from quantdata import env, SubDir
        np_sample  = PrettyObject( env.locations.training.read("grid_sanos_four_samples", fmt=SubDir.PYTREE_HDF5, raise_on_error=True ) )
        """
        num_samples = len(data.dates)
        slc_samples = slice(num_samples-5,num_samples-1) # 3 samples
        np_sample = PrettyObject({ k: np.copy(v) for k,v in TEST_DATA_FOUR_SAMPLES.items() })
            strikes     = data.pure_strike[slc_samples,:],
            is_valid    = data.is_valid[slc_samples,:],
            tte_y       = data.tte_y[slc_samples,:],
            bid         = data.bid_pure_price[slc_samples,:],
            ask         = data.ask_pure_price[slc_samples,:],
            tte         = data.tte_y[slc_samples,:],
            bid_vol     = data.bid_vol[slc_samples,:],
            ask_vol     = data.ask_vol[slc_samples,:]
        """
        
        np_opt_interp  = grid.time_interpolation_weights( np_sample.tte_y )
    
        # Fit
        # ---
        # Works in NP only, and not in grid mode

        np_sample.fwd_vols = np.full( (grid.T,), 0.05, dtype=np.float64 )

        T, N  = grid.TN
        M     = np_sample.strikes.shape[-1] 
        np_sample.dlvsqs = np.zeros((np_sample.strikes.shape[0],T,N-2))
        np_sample.fit_prices = np.zeros((np_sample.strikes.shape[0],M))
        np_sample.densities = np.zeros((np_sample.strikes.shape[0],T,N))

        for isample in range(np_sample.strikes.shape[0]):
            np_sample.dlvsqs[isample,:], \
            np_sample.fit_prices[isample,:], \
            np_sample.densities[isample,:] = \
                fitter( fwd_vols     = np_sample.fwd_vols,
                        opt_strikes  = np_sample.strikes[isample,:],
                        opt_bid      = np_sample.bid[isample,:],
                        opt_ask      = np_sample.ask[isample,:],
                        opt_tte      = np_sample.tte[isample,:],
                        opt_is_calls = None,
                        opt_mask     = np_sample.is_valid[isample,:],
                        return_only_dlvsqs = False,
                        verbose      = Context.all ) 
        np_sample.fit_vols    = bs.implied(np_sample.strikes, np_sample.fit_prices, sqrtT = np.sqrt(np_sample.tte_y) )
        tr_sample             = PrettyObject({ k: bbtr.to_array(v) for k,v in np_sample.items() })
        tr_opt_interp         = { k: bbtr.to_array(v) for k,v in np_opt_interp.items() }

        # Reprice (numpy)
        # ---------------

        np_sanos     = GridSanos( grid, bb = bbnp )
        np_sample.test_density = np_sanos.densities( np_sample.dlvsqs )
        np_sample.test_prices  = np_sanos.price(
                            np_sample.fwd_vols[None,:],
                            np_sample.test_density,
                            opt_strikes = np_sample.strikes,
                            opt_tte     = np_sample.tte,
                            opt_mask    = np_sample.is_valid,
                            opt_is_call = None,
                            opt_interp  = np_opt_interp
        )
        np_sample.test_vols    = bs.implied(np_sample.strikes, np_sample.test_prices, sqrtT = np.sqrt(np_sample.tte_y) )

        # Torch
        # -----

        tr_sanos     = GridSanos( grid, bb = bbtr )
        tr_sample.test_density = tr_sanos.densities( tr_sample.dlvsqs )
        tr_sample.test_prices  = tr_sanos.price(
                            tr_sample.fwd_vols[None,:],
                            tr_sample.test_density,
                            opt_strikes = tr_sample.strikes,
                            opt_tte     = tr_sample.tte,
                            opt_mask    = tr_sample.is_valid,
                            opt_is_call = None,
                            opt_interp  = tr_opt_interp
        )
        tr_sample.test_vols    = bbtr.to_array(bs.implied(
                            tr_sample.strikes.cpu().numpy(),
                            tr_sample.test_prices.cpu().numpy(),
                            sqrtT = np.sqrt(tr_sample.tte_y.cpu().numpy())
        ))

        # validate
        # --------

        assert set(np_sample) == set(tr_sample), f"Expected same keys in numpy and torch samples, found {set(np_sample)} and {set(tr_sample)}"
        for k in np_sample:
            v_np = np_sample[k]
            v_tr = tr_sample[k].cpu().numpy()
            assert np.allclose( v_np, v_tr, atol=1e-6 ), f"Expected numpy and torch values to match for key '{k}', max abs diff {np.max(np.abs(v_np - v_tr))}"
        print("Everything worked!")
