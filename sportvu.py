import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from collections import namedtuple
import pandas as pd

# Define a named tuple for the bucket location
Location = namedtuple('Location', ['X', 'Y', 'Z'])

# Define the locations
# All measurements in feet
BUCKET_HOME = Location(5.25, 25, 10)
BUCKET_AWAY = Location(88.75, 25, 10)
HALF_COURT = 45
THRESHOLD = 4

# Read in the SportVU tracking data
sportvu = []

# fun "is_shot" -- get euclidean distance of ball to hoop 
def is_shot(Ball):
    euclid_distance = 20
    if Ball.X < HALF_COURT:
        euclid_distance = math.sqrt((Ball.X - BUCKET_HOME.X)**2 
                                    + (Ball.Y - BUCKET_HOME.Y)**2 + (Ball.Z - BUCKET_HOME.Z)**2)
    else:
        euclid_distance = math.sqrt((Ball.X - BUCKET_AWAY.X)**2 
                                    + (Ball.Y - BUCKET_AWAY.Y)**2 + (Ball.Z - BUCKET_AWAY.Z)**2)
    return euclid_distance <= 4

# 
def seconds_since_start(quarter, seconds_left):
    return ((quarter - 1) * 720) + (720 - seconds_left)


with open('0021500495.json', 'r') as file:
    all_data = json.load(file)

# These are the two arrays that you need to populate with actual data
shot_times_the_list = []
shot_facts_the_list = []

i = 0
for event in all_data["events"]:
    for moment in event["moments"]:
        ball_info = moment[5][0]
        ball_location = Location(ball_info[2], ball_info[3], ball_info[4])
        if is_shot(ball_location):
            seconds = seconds_since_start(quarter=moment[0], seconds_left=moment[2])
            shot_times_the_list.append(seconds)
            shot_facts_the_list.append(i)
            i += 1


shot_times = np.array(shot_times_the_list) # Between 0 and 2880
shot_facts = np.array(shot_facts_the_list) # Scaled between 0 and 10

# Now that we have our shot times, let's figure out our accuracy and sensitivity
#   Confusion matrix (according to event dataset)
# freethrowy poo

#############################################################################
# Part II: Validation
# We have our predictions for shots, now we want to find
# How accurate our predictions really are. Using the dataset with
# "true" recordings of actions (although we aren't sure about the validity)
# of the dataset, we'll calculate our True Positive and False Negative
# rates.
#############################################################################

###########################################################
## Combine the description columns into 1 column: ACTIONS #
###########################################################

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

###################################################################################
## Convert the PCTIMESTRING column into ELAPSED_SECS (seconds since start) column #
###################################################################################
# Gotta be a more efficient way to do this, but we're taking the PCTIMESTRING column,
#   splitting into a minutes and seconds column (which both represent time until end
#   of the quarter, which is what you see on TV). Then we use our seconds_since_start
#   function to calculate the time elapsed since beginning of the game.
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

###################################################################################
## Filter the dataframe for actions that are shots (either misses or makes) 
# --> Keeping any row which has "Shot", "Miss", or "Free throw" in it                  
###################################################################################
shots_df = event_data[event_data['ACTIONS'].str.contains('Shot|MISS|Free Throw|Jumper|PTS')]
shots_df.to_csv("shot_data.csv", index=False)

def find_unique_elements(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)
    
    # Elements of list a that are not in list b
    unique_in_a = set_a - set_b
    
    # Elements of list b that are not in list a
    unique_in_b = set_b - set_a
    
    return unique_in_a, unique_in_b

false_positives, shots_not_captures = find_unique_elements(shot_times, np.array(shots_df["ELAPSED_SECS"]))
print(false_positives, shots_not_captures)


# # This code creates the timeline display from the shot_times
# # and shot_facts arrays.
# # DO NOT MODIFY THIS CODE APART FROM THE SHOT FACT LABEL
# fig, ax = plt.subplots(figsize=(12,3))
# fig.canvas.manager.set_window_title('Shot Timeline')

# plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
# plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black', width=5, label='shot fact') # <- This is the label you can modify

# ax.spines['bottom'].set_position('zero')
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_color('none')
# ax.tick_params(axis='x', length=20)
# ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0,720,1440,2160,2880])) 
# ax.set_yticks([])

# _, xmax = ax.get_xlim()
# ymin, ymax = ax.get_ylim()
# ax.set_xlim(-15, xmax)
# ax.set_ylim(ymin, ymax+5)
# ax.text(xmax, 2, "time", ha='right', va='top', size=10)
# plt.legend(ncol=5, loc='upper left')

# plt.tight_layout()
# plt.show()

#plt.savefig("Shot_Timeline.png")
