"""
Static Model Visualization
"""

import numpy as np
from cdxcore.dynaplot import FigStore, colors_tableau, color_tableau, figure, m_o_m, quantile_m_o_m
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from cdxcore.subdir import SubDir
from cdxcore.util import fmt_seconds, fmt_datetime
from dataclasses import dataclass

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_train import _cache_root, MetricTracker
from datetime import datetime, date
import torch

@dataclass(frozen=True)
class SummaryVizConfig:
    pass

class SummaryGroup(object):

    def __init__(self, *, name, market, viz_config : SummaryVizConfig = SummaryVizConfig() ):
        self.name       = name
        self.market     = market
        self.models     = {}
        self.viz_config = viz_config
        self.colors     = colors_tableau()
        self.store      = FigStore()
        self.path       = _cache_root(name,ext="pt")
        self.fig        = figure(f"Summary {self.market.grid_market_def.ticker} group '{self.name}'", col_size=8, row_size=6, columns=None )
        self.updated    = datetime.now()

        self.ax_Tloss    = self.fig.add_subplot("Batch Loss")
        self.ax_Tloss.set_xlabel("Epoch")    
        self.ax_Tloss.set_ylabel("Log-Loss")
        self.ax_Tloss.set_yscale("log")
        self.ax_Vloss    = self.fig.add_subplot("Full Loss")
        self.ax_Vloss.set_xlabel("Epoch")    
        self.ax_Vloss.set_ylabel("Log-Loss")
        self.ax_Vloss.set_yscale("log")
        self.ax_Vloss.sharey(self.ax_Tloss)
        self.ax_Vloss.sharex(self.ax_Tloss)

        self.ax_mean_error   = self.fig.add_subplot("Mean of wgt*(fit-market)[wgt>0]")
        self.ax_mean_error.set_xlabel("Date")
        self.ax_mean_error.set_ylabel("Error")
        self.ax_mean_error.tick_params(axis="x", labelrotation=90)

        self.ax_med_error   = self.fig.add_subplot("Median of wgt*(fit-market)[wgt>0]")
        self.ax_med_error.set_xlabel("Date")
        self.ax_med_error.set_ylabel("Error")
        self.ax_med_error.tick_params(axis="x", labelrotation=90)
        self.ax_med_error.sharey(self.ax_mean_error)
        self.ax_med_error.sharex(self.ax_mean_error)

        self.fig.render()

    def update(self, removed : bool):
        if removed:
            self.fig.suptitle(f"Summary {self.market.grid_market_def.ticker} group '{self.name}' -- no longer on disk") 
            self.fig.render()
            return

        self.store.clear()

        # find all files
        models = self.path.files()
        for imdl, model in enumerate(models):
            cm      = self.models.get(model, None)
            if cm is None:
                cm = PrettyObject(  color=next(self.colors),
                                    ffn=self.path.full_file_name(model),
                                    updated=None,
                                    deleted=False,
                                    data=PrettyObject() )

                self.models[model] = cm

        # load where what can
        for model, cm in self.models.items():
            if not self.path.exists(cm.ffn):
                cm.deleted = True
                continue
            cm.deleted = False
            try:
                cm_data     = torch.load(cm.ffn, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Failed to load model {cm.label} from {cm.ffn}: {e}")
                continue

            cm.updated = self.path.get_last_modification_time(model)
            cm.data.tracker = MetricTracker(**cm_data.pop("tracker"))
            cm.data.raw_error_stats = PrettyObject( cm_data.pop("raw_error_stats"))

        # draw
        dates = self.market.dates
        max_q = 1E-6
        self.updated = datetime.now()
        for model, cm in self.models.items():

            if cm.updated is None:
                print(f"Warning: {self.name} {cm.label} has has never been read -- skipping")
                continue

            str_updated     = fmt_seconds( (self.updated - cm.updated).total_seconds() ) if not cm.deleted else "(deleted)"
            tracker         = cm.data.tracker
            raw_error_stats = cm.data.raw_error_stats
            assert len(tracker.trn_epochs) == len(tracker.trn_losses), f"Expected trn_epochs and trn_losses to have the same shape, but found {len(tracker.trn_epochs)} and {len(tracker.trn_losses)}"
            assert len(tracker.val_epochs) == len(tracker.val_losses), f"Expected val_epochs and val_losses to have the same shape, but found {len(tracker.val_epochs)} and {len(tracker.val_losses)}"
            self.store += self.ax_Tloss.plot(tracker.trn_epochs, tracker.trn_losses, "-", color=cm.color, label=f"{model} {str_updated}" )
            self.store += self.ax_Vloss.plot(tracker.val_epochs, tracker.val_losses, "-", color=cm.color, label=f"{model} {str_updated}" )
            self.store += self.ax_Tloss.plot(tracker.trn_epochs[-1:], tracker.trn_losses[-1:], "*", color=cm.color )
            self.store += self.ax_Vloss.plot(tracker.val_epochs[-1:], tracker.val_losses[-1:], "*", color=cm.color )

            assert len(dates) == len(raw_error_stats.means), f"Expected dates and mean_error to have the same length, but found {len(dates)} and {len(raw_error_stats.means)}"
            assert len(dates) == len(raw_error_stats.medians), f"Expected dates and median_error to have the same length, but found {len(dates)} and {len(raw_error_stats.medians)}"
            self.store += self.ax_mean_error.plot(dates, raw_error_stats.means, "-", color=cm.color, label=f"{model} {str_updated}" )
            self.store += self.ax_med_error.plot(dates, raw_error_stats.medians, "-", color=cm.color, label=f"{model} {str_updated}" )
            max_q = max( max_q, np.quantile(raw_error_stats.means, 0.95)*1.1 )
        self.ax_Tloss.legend(loc="upper right")
        self.ax_mean_error.set_ylim(0., max_q)
        self.fig.suptitle(f"Summary {self.market.grid_market_def.ticker} group '{self.name}' [updated {fmt_datetime(self.updated,ignore_ms=True)}]") 
        self.fig.render()
        return max_q

class SummaryViz(object):

    def __init__(self, *, market, viz_config : SummaryVizConfig = SummaryVizConfig() ):
        self.viz_config = viz_config
        self.path       = _cache_root(ext="pt")
        self.groups     = {}
        self.market     = market
  
    def update(self):   
        found_groups          = set( self.path.sub_dirs() )
        for group in sorted(found_groups - set(self.groups)):
            cgroup = self.groups.get(group, None)
            if cgroup is None:
                cgroup = SummaryGroup(name=group, market=self.market, viz_config=self.viz_config)
                self.groups[group] = cgroup
        
        for grpup, grpgfx in self.groups.items():
            grpgfx.update(removed=not found_groups)

