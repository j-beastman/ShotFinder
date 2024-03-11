import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import pandas as pd

# tuple for location in 3D plane
class Location:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

# location consts for court dimensions (UNITS = FEET)
RIM_HEIGHT = 10
HOME_BUCKET = Location(5.25, 25, RIM_HEIGHT)
AWAY_BUCKET = Location(88.75, 25, RIM_HEIGHT)
HALF_COURT = 45
SHOT_DISTANCE_THRESHOLD = 4

# is_shot
# takes Location of ball, determines whether the ball has been shot
# a shot is defined as the ball location being within DISTANCE_THRESHOLD_FOR_SHOT
# feet from the bucket
def is_shot(ball_location: Location):
    distance_from_bucket = 0
    if ball_location.X <= HALF_COURT:
        distance_from_bucket = math.sqrt(((ball_location.X - HOME_BUCKET.X) ** 2)
                                       + ((ball_location.Y - HOME_BUCKET.Y) ** 2)
                                       + ((ball_location.Z - HOME_BUCKET.Z) ** 2))
    else: 
        distance_from_bucket = math.sqrt(((ball_location.X - AWAY_BUCKET.X) ** 2)
                                       + ((ball_location.Y - AWAY_BUCKET.Y) ** 2)
                                       + ((ball_location.Z - AWAY_BUCKET.Z) ** 2))
    return distance_from_bucket <= SHOT_DISTANCE_THRESHOLD

# Used to determine time since start
def seconds_since_start(quarter, seconds_left):
    return ((quarter - 1) * 720) + (720 - seconds_left)

# open sportvu data
with open('0021500495.json', 'r') as file:
    sportvu_data = json.load(file)


shot_times_list = []
QUARTER_IDX = 0
TIMESTAMP_IDX = 1
TIME_LEFT_IN_QUARTER_IDX = 2
MOMENT_SPATIAL_DATA_IDX = 5
BALL_LOCATION_IDX = 0
BALL_MOMENT_X_IDX = 2
BALL_MOMENT_Y_IDX = 3
BALL_MOMENT_Z_IDX = 4

# find all shots in the sportsvu data
for event in sportvu_data["events"]:
    for moment in event["moments"]:
        ball_moment_info = moment[MOMENT_SPATIAL_DATA_IDX][BALL_LOCATION_IDX]
        ball_moment_location = Location(ball_moment_info[BALL_MOMENT_X_IDX], 
                                        ball_moment_info[BALL_MOMENT_Y_IDX], 
                                        ball_moment_info[BALL_MOMENT_Z_IDX])
        if ball_moment_location.Z > RIM_HEIGHT: 
            if is_shot(ball_moment_location):
                seconds = seconds_since_start(quarter=moment[QUARTER_IDX],
                                              seconds_left=moment[TIME_LEFT_IN_QUARTER_IDX])
                shot_times_list.append((seconds, moment[TIMESTAMP_IDX]))

# array of shot_times
# 2 cols: | seconds since start | timestamp (milliseconds)
SHOT_SECONDS_SINCE_START_IDX = 0
SHOT_TIMESTAMP_IDX = 1
shot_times = np.array(shot_times_list)

# TODO: delete
print(shot_times[:20])
print(shot_times.shape)

SHOT_GROUPING_SECONDS_THRESHOLD = 1.5
SHOT_GROUPING_MILLISECONDS_THRESHOLD = SHOT_GROUPING_SECONDS_THRESHOLD * 1000

# scrub shot_times array of duplicates. i.e. save 1 moment for each shot
shot_time_averages = []
i = 0
while i < len(shot_times) - 1:
    # Start a new group with the current timestamp.
    current_group = [shot_times[i][SHOT_SECONDS_SINCE_START_IDX]]
    # Look ahead to the next numbers and add them to the group if they're within the threshold
    #   We compare based on timestamp, not on seconds since start
    while ((i + 1) < len(shot_times)) and abs(shot_times[i + 1][SHOT_TIMESTAMP_IDX] 
                                                - shot_times[i][SHOT_TIMESTAMP_IDX]) <= SHOT_GROUPING_MILLISECONDS_THRESHOLD:
        i += 1
        # Append the seconds since game start
        current_group.append(shot_times[i][SHOT_SECONDS_SINCE_START_IDX])
    
    # Calculate and save only the average of the current group
    avg = int(np.mean(current_group))
    shot_time_averages.append(avg)
    i += 1


# The resulting array of averages
shot_times = np.array(shot_time_averages)
print(shot_times)

print("Found", len(shot_times), "shots")




###############################################################################
#####                         UNIT TESTING                                #####
###############################################################################
# unit tests for is_shot
def is_shot_tests():
    failed_tests = ""
    if not (is_shot(HOME_BUCKET)):
        failed_tests += "in home bucket\n"
        
    if not is_shot(AWAY_BUCKET):
        failed_tests += "in away bucket\n"
    
    if not is_shot(Location((AWAY_BUCKET.X + 1), AWAY_BUCKET.Y, AWAY_BUCKET.Z)):
        failed_tests += "1 x ft from away bucket\n"
    
    if not is_shot(Location(AWAY_BUCKET.X, (AWAY_BUCKET.Y + 1), AWAY_BUCKET.Z)):
        failed_tests += "1 y ft from away bucket\n"
    
    if not is_shot(Location(AWAY_BUCKET.X, AWAY_BUCKET.Y, (AWAY_BUCKET.Z + 1))):
        failed_tests += "1 z ft from away bucket\n"
    
    if is_shot(Location((HOME_BUCKET.X + SHOT_DISTANCE_THRESHOLD + 0.0001), HOME_BUCKET.Y, HOME_BUCKET.Z)):
        failed_tests += "just over threshold x ft from HOME bucket\n"

    if is_shot(Location(HOME_BUCKET.X, (HOME_BUCKET.Y + SHOT_DISTANCE_THRESHOLD + 0.0001), HOME_BUCKET.Z)):
        failed_tests += "just over threshold y ft from HOME bucket\n"
    
    if is_shot(Location(HOME_BUCKET.X, HOME_BUCKET.Y, (HOME_BUCKET.Z + SHOT_DISTANCE_THRESHOLD + 0.0001))):
        failed_tests += "just over threshold z ft from HOME bucket\n"

    if is_shot(Location((AWAY_BUCKET.X + SHOT_DISTANCE_THRESHOLD + 0.0001), AWAY_BUCKET.Y, AWAY_BUCKET.Z)):
        failed_tests += "just over threshold x ft from away bucket\n"

    if is_shot(Location(AWAY_BUCKET.X, (AWAY_BUCKET.Y + SHOT_DISTANCE_THRESHOLD + 0.0001), AWAY_BUCKET.Z)):
        failed_tests += "just over threshold y ft from away bucket\n"
    
    if is_shot(Location(AWAY_BUCKET.X, AWAY_BUCKET.Y, (AWAY_BUCKET.Z + SHOT_DISTANCE_THRESHOLD + 0.0001))):
        failed_tests += "just over threshold z ft from away bucket\n"
    
    if not is_shot(Location((HOME_BUCKET.X - SHOT_DISTANCE_THRESHOLD + 0.0001), HOME_BUCKET.Y, HOME_BUCKET.Z)):
        failed_tests += "just under threshold x ft from HOME bucket\n"

    if not is_shot(Location(HOME_BUCKET.X, (HOME_BUCKET.Y - SHOT_DISTANCE_THRESHOLD + 0.0001), HOME_BUCKET.Z)):
        failed_tests += "just under threshold y ft from HOME bucket\n"
    
    if not is_shot(Location(HOME_BUCKET.X, HOME_BUCKET.Y, (HOME_BUCKET.Z - SHOT_DISTANCE_THRESHOLD + 0.0001))):
        failed_tests += "just under threshold z ft from HOME bucket\n"

    if not is_shot(Location((AWAY_BUCKET.X - SHOT_DISTANCE_THRESHOLD + 0.0001), AWAY_BUCKET.Y, AWAY_BUCKET.Z)):
        failed_tests += "just under threshold x ft from away bucket\n"

    if not is_shot(Location(AWAY_BUCKET.X, (AWAY_BUCKET.Y - SHOT_DISTANCE_THRESHOLD + 0.0001), AWAY_BUCKET.Z)):
        failed_tests += "just under threshold y ft from away bucket\n"
    
    if not is_shot(Location(AWAY_BUCKET.X, AWAY_BUCKET.Y, (AWAY_BUCKET.Z - SHOT_DISTANCE_THRESHOLD + 0.0001))):
        failed_tests += "just under threshold z ft from away bucket\n"
    

    print("failed tests for is_shot:\n", failed_tests)

# unit tests for seconds_since_start
def seconds_since_start_tests():
    failed_tests = ""
    if not (seconds_since_start(1, 720) == 0):
        failed_tests += "start of 1st quarter\n"
    
    if not (seconds_since_start(1, 0) == 720):
        failed_tests += "end of 1st quarter\n"
    
    if not (seconds_since_start(2, 720) == 720):
        failed_tests += "start of 2nd quarter\n"
    
    if not (seconds_since_start(2, 0) == (720 * 2)):
        failed_tests += "end of 2nd quarter\n"
    
    if not (seconds_since_start(3, 720) == (720 * 2)):
        failed_tests += "start of 3rd quarter\n"
    
    if not (seconds_since_start(3, 0) == (720 * 3)):
        failed_tests += "end of 3rd quarter\n"
    
    if not (seconds_since_start(4, 720) == (720 * 3)):
        failed_tests += "start of 4th quarter\n"
    
    if not (seconds_since_start(4, 0) == (720 * 4)):
        failed_tests += "end of 4th quarter\n"

    print("failed tests for seconds_since_start:\n", failed_tests)

# UNCOMMENT TO SEE UNIT TESTS:
# print("UNIT TEST RESULTS:\n")
# is_shot_tests()
# seconds_since_start_tests()