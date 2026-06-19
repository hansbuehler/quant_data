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

class SummaryViz(object):

    PATTERNS = ["-",".:","--",".:"]

    def __init__(self, market : PrettyObject, path : SubDir, *, viz_config : VizConfig ):
        self.market     = market
        self.viz_config = viz_config
        self.store      = FigStore()
        self.path       = path(ext="pt")
        self.fig        = figure(f"Summary {self.market.grid_market_def.ticker}", col_size=8, row_size=4, columns=None )

        self.ax_Tloss    = self.fig.add_subplot("Loss")
        self.ax_Tloss.set_xlabel("Epoch")    
        self.ax_Tloss.set_ylabel("Log-Loss")
        self.ax_Tloss.set_yscale("log")
        self.ax_Vloss    = self.fig.add_subplot("Loss")
        self.ax_Vloss.set_xlabel("Epoch")    
        self.ax_Vloss.set_ylabel("Log-Loss")
        self.ax_Vloss.set_yscale("log")
        self.ax_Vloss.sharey(self.ax_Tloss)
        self.ax_Vloss.sharex(self.ax_Tloss)

        self.ax_mean_error   = self.fig.add_subplot("Mean and Median of err[wgt>0] divided by mean(wgt[wgt>0])")
        self.ax_mean_error.set_xlabel("Date")
        self.ax_mean_error.set_ylabel("Error")
        self.ax_mean_error.tick_params(axis="x", labelrotation=90)

        self.ax_med_error   = self.fig.add_subplot("Mean and Median of err[wgt>0] divided by mean(wgt[wgt>0])")
        self.ax_med_error.set_xlabel("Date")
        self.ax_med_error.set_ylabel("Error")
        self.ax_med_error.tick_params(axis="x", labelrotation=90)
        self.ax_med_error.sharey(self.ax_mean_error)
        self.ax_med_error.sharex(self.ax_mean_error)

        self.fig.render()
        self.patterns = list( SummaryViz.PATTERNS )
        self.classes  = {}
        self.models   = {}

    class SummaryModel(object):

        def __init__(self, *, name, pattern ):
            self.name    = name
            self.pattern = pattern
            self.colors  = colors_tableau()
            self.models  = {}

        def scan(self, *, cpath : SubDir ):
            models = cpath.sub_dirs()
            for imdl, model in enumerate(models):
                cmpath  = cpath(model)
                cm      = self.models.get(model, None)
                if cm is None:
                    cm = PrettyObject( label=f"{self.name} {model}",
                                       color=next(self.colors),
                                       pattern=self.pattern,
                                       ffn=cm.cmpath.full_file_name("last"),
                                       updated=None )
                    self.models[model] = cm
                cm.updated = cmpath.last_modified("last")
            self.updated = datetime.now()

        def draw( self, *, ax_loss, ax_error, store, market ):
        
            dates = market.dates

            for model, cm in self.models.items():
                cm_data     = torch.load(cm.ffn)
                tracker     = MetricTracker(**cm_data.pop("tracker"))
                error_stats = PrettyObject( cm_data.pop("error_stats"))
                assert tracker.trn_epochs.shape == tracker.trn_losses.shape, f"Expected trn_epochs and trn_losses to have the same shape, but found {tracker.trn_epochs.shape} and {tracker.trn_losses.shape}"
                assert tracker.val_epochs.shape == tracker.val_losses.shape, f"Expected val_epochs and val_losses to have the same shape, but found {tracker.val_epochs.shape} and {tracker.val_losses.shape}"
                store += ax_Tloss.plot(tracker.trn_epochs, tracker.trn_losses, cm.pattern, color=cm.color, label=cm.label )
                store += ax_Vloss.plot(tracker.val_epochs, tracker.val_losses, cm.pattern, color=cm.color, label=cm.label )

                assert len(dates) == len(error_stats.mean_error), f"Expected dates and mean_error to have the same length, but found {len(dates)} and {len(error_stats.mean_error)}"
                assert len(dates) == len(error_stats.median_error), f"Expected dates and median_error to have the same length, but found {len(dates)} and {len(error_stats.median_error)}"
                store += ax_mean_error.plot(dates, error_stats.mean_error, cm.pattern, color=cm.color, label=cm.label )
                store += ax_med_error.plot(dates, error_stats.median_error, cm.pattern, color=cm.color, label=cm.label )

            ax_Tloss.legend(loc="upper right")

    def update(self):

        # identify all classes
        # --------------------

        classes     = path.sub_dirs()
        for iclss, clss in enumerate(classes):
            cmodels = self.classes.get(clss, None)
            if cmodels is None:
                cmodels = SummaryModel( name=clss, pattern=self.patterns.pop(0) )
                self.classes[clss] = cmodels
                if self.patterns == []:
                    self.patterns = list( SummaryViz.PATTERNS )

            # identify all models per class
            # -----------------------------

            self.store.clear()

            for cls, cmodels in self.classes.items():                
                cmodels.scan( cpath=path(cls) )
                cmodels.draw( ax_loss=self.ax_loss, ax_error=self.ax_error, store=self.store, market=self.market )
                
            self.fig.render()





        




