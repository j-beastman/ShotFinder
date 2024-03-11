import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import pandas as pd

# Define a named tuple for the bucket location
# Location = namedtuple('Location', ['X', 'Y', 'Z'])
class Location:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def distance_to(self, other):
        return math.sqrt((self.X - other.X) ** 2 + (self.Y - other.Y) ** 2)


# Define the locations
# All measurements in feet
BUCKET_HOME = Location(5.25, 25, 10)
BUCKET_AWAY = Location(88.75, 25, 10)
HALF_COURT = 45
DISTANCE_THRESHOLD_FOR_SHOT = 2

def euclid_distance(object1 : Location, object2 : Location):
    return math.sqrt((object1.X - object2.X) ** 2 + (object1.Y - object2.Y) ** 2)

# fun "is_shot" -- get euclidean distance of ball to hoop 
def is_shot(Ball):
    euclid_distance:float = 20.0
    if Ball.X <= HALF_COURT:
        euclid_distance = math.sqrt(((Ball.X - BUCKET_HOME.X) ** 2)
                                    + ((Ball.Y - BUCKET_HOME.Y) ** 2) + ((Ball.Z - BUCKET_HOME.Z) ** 2))
    else:
        euclid_distance = math.sqrt(((Ball.X - BUCKET_AWAY.X) ** 2)
                                    + ((Ball.Y - BUCKET_AWAY.Y) ** 2) + ((Ball.Z - BUCKET_AWAY.Z) ** 2))
    return euclid_distance <= DISTANCE_THRESHOLD_FOR_SHOT

# Used to determine time since start
def seconds_since_start(quarter, seconds_left):
    return ((quarter - 1) * 720) + (720 - seconds_left)

####################################################################################
##  Convert raw data into something more usable, only use once
####################################################################################
def clean_data():
    # Open up the SportsVu data
    with open('0021500495.json', 'r') as file:
        sportVU = json.load(file)

    all_moments = []

    # Iterate through each event
    for event in sportVU['events']:
        # Iterate through each moment in the event and extend the all_moments list
        for moment in event['moments']:
            if moment not in all_moments:
                all_moments.append(moment)

    # Specify the file path where you want to save the JSON file
    file_path = "output.json"

    # Write the list to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(all_moments, json_file)


# Now, we just read from our moments json file (the cleaned up one)
with open("output.json", 'r') as json_file:
    all_moments = json.load(json_file)


shot_times_list = []
skip_to_index = 0
index_of_list = -1

####################################################################################
# Iterate through each moment, if 
####################################################################################
for index, moment in enumerate(all_moments):
    if index < skip_to_index:
        continue
    skip_to_index = index
    ball_info = moment[5][0]
    ball_location = Location(ball_info[2], ball_info[3], ball_info[4])
    # If the ball is above the rim, it has the potential to be a shot.
    if ball_location.Z > 10:
        if is_shot(ball_location):
            seconds = seconds_since_start(quarter=moment[0], seconds_left=moment[2])
            timestamp = moment[1]
            # If shot list is empty, we have to add to list
            if len(shot_times_list) == 0:
                shot_times_list.append((seconds, timestamp))
            # If the timestamp of the current shot is within 0.5 second (500 milliseconds)
            #   of any other shots recorded in our shot_times array,
            #   then we don't want to add to our shot_times array, this solves a bug that is 
            #   present in the last ~10 shots--comment out and see what happens.
            elif any(timestamp == shot_time[1] for shot_time in shot_times_list) or (
                                    abs(shot_times_list[index_of_list][1] - timestamp) < 500):
                continue
            else:
                shot_times_list.append((seconds, timestamp))
                index_of_list += 1
                inside_shot_threshold = True
                while inside_shot_threshold:
                    # Grab new moment
                    skip_to_index += 1
                    new_moment = all_moments[skip_to_index]
                    new_ball_info = new_moment[5][0]
                    new_ball_location = Location(new_ball_info[2], new_ball_info[3], new_ball_info[4])
                    # If it is not a shot, we wanna get outta this loop
                    if is_shot(new_ball_location):
                        skip_to_index += 1
                    else:
                        inside_shot_threshold = False



########################################
#### How good was our program? #########
########################################
shot_times = np.array(shot_times_list)
print("Found", len(shot_times), "shots")
TRUE_SHOT_TOTAL = 231 # according to sportsreference
print("Failed to find", (TRUE_SHOT_TOTAL - len(shot_times)), "shots!")

# Code to print out shots (game time, realtime, from our array
testing = shot_times[:-5]
for i, item in enumerate(shot_times):
    for value in item:
        print("{:.1f}".format(value), end=' ')
    print("Shot number", i)

shot_facts = np.arange(0, len(shot_times), 1)
arc_lengths = shot_facts


####################################################################################
##  Part II: Calculate arc length of shot vs. distance to basketball        
####################################################################################
def ball_in_air(moment, ball_location, threshold=3):
    # Loc_info is info of ball + players
    loc_info = moment[5]
    # Loop through all players, if any are within 'threshold' feet of ball, 
    try:
        for i in range(1, 10):
            print(i)
            player_info = loc_info[i]
            player_location = Location(player_info[2], player_info[3], player_info[4])
            if euclid_distance(player_location, ball_location) < threshold:
                return False
    except IndexError:
        print("This moment is messed up")
    return True

def get_ball_location(moment):
    ball_info = moment[5][0]
    return Location(ball_info[2], ball_info[3], ball_info[4])

distance = []
shots_found = 0
index_in_all_moments = 0
timestamps = [shot[1] for shot in shot_times]
index = 0
for timestamp in timestamps:
    timestamp_not_found = True
    # Iterate through moments array until we find the timestamp
    while timestamp_not_found:
        # Did we find the moment when the shot occurred?
        if timestamp == all_moments[index][1]:
            timestamp_not_found = False
        else:
            index += 1
    moment_of_shot = all_moments[index]
    ball_location = get_ball_location(moment_of_shot)

    ball_arc_information = []
    ball_arc_information.append(ball_location)
    in_air = True
    index_to_reverse = 1
    # Go back in time until the ball is the hands of a player.
    while in_air:
        past_moment = all_moments[index - index_to_reverse]
        past_ball_location = get_ball_location(past_moment)
        in_air:bool = ball_in_air(past_moment, past_ball_location)
        index_to_reverse += 1
    ball_arc_information.append(past_ball_location)
    distance.append(ball_arc_information)


# List to store distances
distances = []

# Calculate distances for each tuple of Location objects
for location_tuple in distance:
    distance = location_tuple[0].distance_to(location_tuple[1])
    distances.append(distance)

filtered_distances = [d for d in distances if d <= 30]
def get_bucket(number):
    # Align range starting from 0 and find the bucket
    bucket_index = (number - 1) // 3
    return bucket_index + 1  # Return the bucket number (1-based indexing)

# Example usage:
buckets = [get_bucket(number) for number in filtered_distances]
scaled_distances = buckets

# Find indices of elements larger than 30
indices = [index for index, value in enumerate(distances) if value < 30]

print("Indices of elements larger than 30:", indices)

# Map through the shot_times array to get the shot times in relation to game clock
#   rather than the timestamp
shot_times = list(map(lambda x: x[0], shot_times))
shot_times = [shot_times[i] for i in indices]
####################################################################################
# Below is given code          
###################################################################################
# This code creates the timeline display from the shot_times
# and shot_facts arrays.
# DO NOT MODIFY THIS CODE APART FROM THE SHOT FACT LABEL
fig, ax = plt.subplots(figsize=(12,3))
fig.canvas.manager.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, scaled_distances, bottom=2, color='royalblue', edgecolor='black', width=5, label='shot fact') # <- This is the label you can modify

ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.tick_params(axis='x', length=20)
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0,720,1440,2160,2880])) 
ax.set_yticks([])

_, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(-15, xmax)
ax.set_ylim(ymin, ymax+5)
ax.text(xmax, 2, "time", ha='right', va='top', size=10)
plt.legend(ncol=5, loc='upper left')

plt.tight_layout()
plt.show()

plt.savefig("Shot_Timeline.png")

###################################################################################
# Below is omitted code which we used to figure out our false positives/negatives
#   we figured out that sportsreference data is not super reliable, so we decided
#   we couldn't trust the results we gathered from the code below.           
###################################################################################

#############################################################################
# Validation:
# We have our predictions for shots, now we want to find
# How accurate our predictions really are. Using the dataset with
# "true" recordings of actions (although we aren't sure about the validity)
# of the dataset, we'll calculate our True Positive and False Negative
# rates.
#############################################################################

###########################################################
## Combine the description columns into 1 column: ACTIONS #
###########################################################
def omitted1():
    event_data = pd.read_csv("0021500495.csv")
    # do .copy() to please pandas
    event_data_home = event_data[event_data["HOMEDESCRIPTION"].notna()].copy()
    event_data_away = event_data[event_data["VISITORDESCRIPTION"].notna()].copy()

    event_data_home.drop(columns=["VISITORDESCRIPTION"])
    event_data_away.drop(columns=["HOMEDESCRIPTION"])

    event_data_home["ACTIONS"] = event_data_home["HOMEDESCRIPTION"]
    event_data_away["ACTIONS"] = event_data_away["VISITORDESCRIPTION"]

    event_data_home = event_data_home.drop(columns=["HOMEDESCRIPTION"])
    event_data_away = event_data_away.drop(columns=["VISITORDESCRIPTION"])

    event_data = pd.concat([event_data_home, event_data_away], axis=0)

    event_data.reset_index(inplace=True, drop=True)

    event_data = event_data[['PERIOD', 'PCTIMESTRING', 'ACTIONS']]

    return event_data

###################################################################################
## Convert the PCTIMESTRING column into ELAPSED_SECS (seconds since start) column #
###################################################################################
# Gotta be a more efficient way to do this, but we're taking the PCTIMESTRING column,
#   splitting into a minutes and seconds column (which both represent time until end
#   of the quarter, which is what you see on TV). Then we use our seconds_since_start
#   function to calculate the time elapsed since beginning of the game.
def omitted3(event_data):
    min_sec_split = event_data["PCTIMESTRING"].str.split(":", expand=True).astype(int)
    minutes_till = min_sec_split[0]
    seconds_till = min_sec_split[1]
    total_seconds = minutes_till * 60 + seconds_till
    seconds = total_seconds.combine(event_data["PERIOD"], 
                                        func=lambda seconds_left, 
                                                    quarter: seconds_since_start(quarter, seconds_left))
    # Take new calculations, put into elapsed time column, drop the PCTIMESTRING column
    event_data['ELAPSED_SECS'] = seconds
    event_data = event_data[['PERIOD', 'ELAPSED_SECS', 'ACTIONS']]
    
    event_data.to_csv("event_data.csv", index=False)
    return event_data

###################################################################################
## Filter the dataframe for actions that are shots (either misses or makes) 
# --> Keeping any row which has "Shot", "Miss", or "Free throw" in it                  
###################################################################################
def omitted2(event_data):
    shots_df = event_data[event_data['ACTIONS'].str.contains('Shot|MISS|Free Throw|Jumper|PTS')]
    shots_df.to_csv("shot_data.csv", index=False)
    print(shots_df.shape, "Total shots according to this df")

    def find_approx_matches(list_a, list_b, tolerance=3):
        # Convert both lists to numpy arrays for efficient computation
        import numpy as np
        array_a = np.array(list_a)
        array_b = np.array(list_b)

        # Initialize sets to keep track of indices with matches
        matches_in_a = set()
        matches_in_b = set()

        # Check each element in array_a for close matches in array_b
        for i, value_a in enumerate(array_a):
            # Compute the absolute difference with all elements in array_b
            diffs = np.abs(array_b - value_a)
            # Find indices in array_b where the difference is within the tolerance
            close_indices = np.where(diffs <= tolerance)[0]
            print(close_indices, "\n")
            if len(close_indices) > 0:
                # If there are close matches, record the indices
                matches_in_a.add(i)
                matches_in_b.update(close_indices)

        # Determine elements that were not matched
        unmatched_in_a = set(range(len(array_a))) - matches_in_a
        unmatched_in_b = set(range(len(array_b))) - matches_in_b

        # Convert indices back to original elements (optional, depending on needs)
        unmatched_elements_a = array_a[list(unmatched_in_a)]
        unmatched_elements_b = array_b[list(unmatched_in_b)]

        return unmatched_elements_a, unmatched_elements_b

    # Assuming shot_times and shots_df["ELAPSED_SECS"] are defined
    # Convert shots_df["ELAPSED_SECS"] to a list if it's not already
    shots_df_elapsed_secs_list = shots_df["ELAPSED_SECS"].tolist()

    # Find unmatched elements with the updated criteria
    # Shot_times is sportsvu data
    false_positives, shots_not_captured = find_approx_matches(shot_times, shots_df_elapsed_secs_list, tolerance=3)

    print("Found", len(false_positives), "shots from data that are not close to any in sportsreference")
    print("Failed to find", len(shots_not_captured), "shots according to sportsreference that are close to any in data")

# event_data = omitted1()
# event_data = omitted3(event_data)
# omitted2(event_data)