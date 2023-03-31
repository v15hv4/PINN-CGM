import pandas as pd
from utils import dt2int


class C3RData:
    def __init__(
        self,
        data_dir,
        cgm_data="MonitorCGM.txt",
        meal_data="MonitorMeal.txt",
        insulin_data="MonitorTotalBolus.txt",
    ):
        self.data_dir = data_dir
        self.cgm_data = pd.read_csv(f"{data_dir}/{cgm_data}", sep="|")
        self.meal_data = pd.read_csv(f"{data_dir}/{meal_data}", sep="|")
        self.insulin_data = pd.read_csv(f"{data_dir}/{insulin_data}", sep="|")

    def get_case_df(self, df, deident_id, from_dt, to_dt, dtcol="LocalDtTm"):
        df_case = df.copy()
        df_case = df_case.loc[df_case.DeidentID == deident_id]
        df_case[dtcol] = df_case[dtcol].map(dt2int)
        df_case = df_case.sort_values(by=[dtcol]).reset_index()
        df_case = df_case.loc[(df_case[dtcol] >= from_dt) & (df_case[dtcol] < to_dt)]

        return df_case

    def get_dates(self, deident_id):
        cgm_df = self.cgm_data
        cgm_dates = (
            cgm_df["LocalDtTm"]
            .map(lambda dt: dt[:10])
            .loc[cgm_df["DeidentID"] == deident_id]
            .sort_values()
            .unique()
        )

        return cgm_dates

    # get data of a given case
    def get_case(self, deident_id, from_dt, to_dt):
        from_dt = dt2int(from_dt)
        to_dt = dt2int(to_dt)

        cgm = self.get_case_df(self.cgm_data, deident_id, from_dt, to_dt)
        meal = self.get_case_df(self.meal_data, deident_id, from_dt, to_dt)
        insulin = self.get_case_df(
            self.insulin_data,
            deident_id,
            from_dt,
            to_dt,
            "LocalDeliveredDtTm",
        )

        return cgm, meal, insulin
