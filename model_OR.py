from pyproj import Proj, Transformer
from tabulate import tabulate
from geopy.distance import geodesic
import pandas as pd
import joblib
import numpy as np
import polars as pl
import math

model_empty_leg_filename = 'V3_model_time_driving_empty_leg.pkl'
model_load_leg_filename = 'V3_model_time_driving_load_leg.pkl'
loaded_model_empty_leg = joblib.load(model_empty_leg_filename)
loaded_model_load_leg = joblib.load(model_load_leg_filename)

rotation_angle = math.radians(35)


class Location:
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

    def __init__(self, Latitude, Longitude, latest_move_start_time=None, leg_type="empty"):
        self.Latitude = Latitude
        self.Longitude = Longitude
        self.x, self.y = Location.transformer.transform(Longitude, Latitude)
        self.leg_type = leg_type  # Default is "empty"

    def predict_travel_time(self, location):
        if self.leg_type == "empty":
            features = {
                'latitude_first': self.Latitude,
                'longitude_first': self.Longitude,
                'latitude_lift': location.Latitude,
                'longitude_lift': location.Longitude
            }

            rotated_x_empty_leg = ((features['longitude_first'] - features['longitude_lift']) * 40075000 * math.cos(math.radians(features['latitude_lift'])) / 360) * math.cos(rotation_angle) - \
                                ((features['latitude_first'] - features['latitude_lift']) * 111320) * math.sin(rotation_angle)

            rotated_y_empty_leg = ((features['longitude_first'] - features['longitude_lift']) * 40075000 * math.cos(math.radians(features['latitude_lift'])) / 360) * math.sin(rotation_angle) + \
                                ((features['latitude_first'] - features['latitude_lift']) * 111320) * math.cos(rotation_angle)

            bearing_empty_leg = (
                (
                    np.degrees(np.arctan2(
                        np.sin(np.radians(features['longitude_lift'] - features['longitude_first'])) * np.cos(np.radians(features['latitude_lift'])),
                        np.cos(np.radians(features['latitude_first'])) * np.sin(np.radians(features['latitude_lift'])) -
                        np.sin(np.radians(features['latitude_first'])) * np.cos(np.radians(features['latitude_lift'])) *
                        np.cos(np.radians(features['longitude_lift'] - features['longitude_first']))
                    )) + 360
                ) % 360
            )

            feature_array = np.array([[features['latitude_first'], features['longitude_first'],
                                    features['latitude_lift'], features['longitude_lift'],
                                    rotated_x_empty_leg, rotated_y_empty_leg, bearing_empty_leg]])

            predicted_time = loaded_model_empty_leg.predict(feature_array)[0]

        else:
            features = {
                'latitude_lift': self.Latitude,
                'longitude_lift': self.Longitude,
                'latitude_last': location.Latitude,
                'longitude_last': location.Longitude
            }

            rotated_x_load_leg = ((features['longitude_last'] - features['longitude_lift']) * 40075000 * math.cos(math.radians(features['latitude_lift'])) / 360) * math.cos(rotation_angle) - \
                                ((features['latitude_last'] - features['latitude_lift']) * 111320) * math.sin(rotation_angle)

            rotated_y_load_leg = ((features['longitude_last'] - features['longitude_lift']) * 40075000 * math.cos(math.radians(features['latitude_lift'])) / 360) * math.sin(rotation_angle) + \
                                ((features['latitude_last'] - features['latitude_lift']) * 111320) * math.cos(rotation_angle)

            bearing_load_leg = (
                (
                    np.degrees(np.arctan2(
                        np.sin(np.radians(features['longitude_last'] - features['longitude_lift'])) * np.cos(np.radians(features['latitude_last'])),
                        np.cos(np.radians(features['latitude_lift'])) * np.sin(np.radians(features['latitude_last'])) -
                        np.sin(np.radians(features['latitude_lift'])) * np.cos(np.radians(features['latitude_last'])) *
                        np.cos(np.radians(features['longitude_last'] - features['longitude_lift']))
                    )) + 360
                ) % 360
            )

            feature_array = np.array([[features['latitude_lift'], features['longitude_lift'],
                                    features['latitude_last'], features['longitude_last'],
                                    rotated_x_load_leg, rotated_y_load_leg, bearing_load_leg]])

            predicted_time = loaded_model_load_leg.predict(feature_array)[0]

        return predicted_time

    def distance_to_man(self, location):
        dx = abs(self.x - location.x)
        dy = abs(self.y - location.y)
        return dx + dy

    def duration(self, location):
        travel_time = self.predict_travel_time(location)
        return float(travel_time)

    def __str__(self):
        return "(" + str(self.Latitude) + "," + str(self.Longitude) + ")"


class SC:
    def __init__(self, name, index, start_position, start_time, real_wis=None):
        self.name = name
        self.start_position = start_position
        self.start_time = start_time
        self.current_position = start_position
        self.last_position = start_position
        self.last_WI_finish_time = None
        self.next_WI = None
        self.last_WI = None
        self.index = index
        self.real_wis = None
        if self.real_wis is None:
            self.real_wis = []

    def __str__(self):
        return self.name


class WI:
    def __init__(self, name, index, kind, start_position, end_position, earliest_move_start_time,
                latest_move_start_time, earliest_move_end_time, latest_move_end_time,
                container_above=None, container_under=None, real_assignment=None):
        self.index = index
        self.name = name
        self.kind = kind
        self.start_position = start_position
        self.end_position = end_position
        self.earliest_move_start_time = earliest_move_start_time
        self.latest_move_start_time = latest_move_start_time
        self.earliest_move_end_time = earliest_move_end_time
        self.latest_move_end_time = latest_move_end_time
        self.current_position = start_position
        self.planned_start_time = None
        self.planned_end_time = None
        self.container_above = container_above
        self.time_when_container_above_moved = -1
        self.container_under = container_under
        self.sc = None
        self.sc_coming_from = None
        self.sc_start_for = None
        self.sc_at_initial_at = None
        self.sc_at_final_at = None
        self.planned_time_is_calculated = False
        self.real_assignment = real_assignment

    def __str__(self):
        return "WI:" + str(self.index)

class Container:
    pass


class Problem:
    def __init__(self, shift_id, planning_interval, wis = None, scs = None):
        if wis is None:
            self.wis = []
        if scs is None:
            self.scs = []
        self.wis_number = None
        self.sc_number = None
        self.shift_id = shift_id
        self.planning_interval = planning_interval
        self.start_time = None
        self.end_time = None
        self.shift_end_time = None
        self.shift_duration = None
        self.real_world_time = 0

    def read_file(self, shift_file, wi_file, SC_num_arbitrary=None):
        df_shift = pd.read_parquet(shift_file)
        df_wi = pd.read_parquet(wi_file)

        df_shift_id = df_shift.loc[df_shift['Shift ID'] == self.shift_id]
        df_shift_id_row = df_shift_id.iloc[0]

        self.start_time = df_shift_id_row['Shift Start']
        self.shift_end_time = df_shift_id_row['Shift End']
        self.shift_duration = (self.shift_end_time - self.start_time).total_seconds() / 60

        if self.planning_interval is None:
            self.end_time = self.shift_end_time
            self.planning_interval = (self.shift_end_time - self.start_time).total_seconds() / 60
        else:
            self.end_time = self.start_time + pd.Timedelta(minutes=self.planning_interval)

        if self.end_time > self.shift_end_time:
            raise ValueError(f"End time {self.end_time} exceeds the shift end time {self.shift_end_time}.")

        scs_info = df_shift_id_row["Carrier Positions"]
        if SC_num_arbitrary is not None and len(scs_info) >= SC_num_arbitrary:
            scs_info = scs_info[:SC_num_arbitrary]

        self.sc_number = len(scs_info)

        for straddle_index, straddles in enumerate(scs_info, start=0):
            sc_name_long = straddles['Carrier Name']
            sc_name_short = sc_name_long.replace('DEHAM_CTH_', '')

            latitude = straddles['Latitude']
            longitude = straddles['Longitude']
            # latitude = straddles['Longitude']
            # longitude = straddles['Latitude']

            self.scs.append(SC(sc_name_short, straddle_index, Location(latitude, longitude), self.start_time))

        df_wi_time_horizon = df_wi[(df_wi['msg|timestamp_first'] >= self.start_time) &
                                   (df_wi['msg|timestamp_first'] <= self.end_time)
                                   ]

        for wi_index, (index, row) in enumerate(df_wi_time_horizon.iterrows()):
            if not (60 <= row["time_driving_empty_leg"] <= 1200 and 60 <= row["time_driving_load_leg"] <= 1200):
                continue
            if not (0 <= row["time_waiting_empty_leg"] <= 600 and 0 <= row["time_waiting_load_leg"] <= 600):
                continue
            name = row["che|@|cycle|@|id"]
            kind = row["cycle_type"]
            start_position = Location(row['latitude_lift'], row['longitude_lift'])
            end_position = Location(row['latitude_last'], row['longitude_last'])
            real_assignment_name = row['che|@|name']
            earliest_move_start_time = row['Earliest Move Start Time']
            latest_move_start_time = row['Latest Move Start Time']
            earliest_move_end_time = row['Earliest Move End Time']
            latest_move_end_time = row['Latest Move End Time']
            sc_assigned = None
            self.real_world_time += row["time_driving_empty_leg"]

            for sc in self.scs:
                if sc.name in real_assignment_name:
                    sc_assigned = sc

            self.wis.append(WI(name, wi_index, kind, start_position, end_position, earliest_move_start_time,
                               latest_move_start_time, earliest_move_end_time, latest_move_end_time,
                               real_assignment=sc_assigned))

        self.wis_number = len(self.wis)

        for wi in self.wis:
            sc = wi.real_assignment
            if sc is not None:
                sc.real_wis.append(wi)

        self.wis = [i for i in self.wis if pd.notna(i.latest_move_start_time)]

        self.wis.sort(key=lambda i: i.latest_move_start_time)

    def report(self):
        data = [
            ['Shift ID', self.shift_id],
            ['Shift Interval', f'{self.start_time} - {self.shift_end_time}'],
            ['Shift Duration', f'{self.shift_duration}'],
            ['Planning Interval', f'{self.start_time} - {self.end_time}'],
            ['Planning Duration', f'{self.planning_interval} mins.'],
            ['SCs Number', self.sc_number],
            ['WI Number', self.wis_number]
        ]
        print(tabulate(data, headers=['Parameter', 'Value'], tablefmt='fancy_grid'), "\n\n")

    def report_latex(self):
        data = [
            ['Shift ID', self.shift_id],
            ['Shift Interval', f'{self.start_time} - {self.shift_end_time}'],
            ['Planning Interval', f'{self.start_time} - {self.end_time}'],
            ['Planning Duration', f'{self.planning_interval} mins.'],
            ['SCs Number', self.sc_number],
            ['WI Number', self.wis_number]
        ]
        print(tabulate(data, headers=['Parameter', 'Value'], tablefmt='latex_booktabs'), "\n\n")


