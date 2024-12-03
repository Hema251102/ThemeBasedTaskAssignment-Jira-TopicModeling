import os
import pymongo
import pandas as pd
from properties import datafolder, mongourl

# Connect to database
client = pymongo.MongoClient(mongourl)
db = client["test"]

# Find issues with assignees, labels, priorities and types
mongo_filter = {'$and': [{'assignee': {'$exists': True, '$not': {'$size': 0}}}, {'labels': {'$exists': True, '$not': {'$size': 0}}},
                         {'priority.id': {'$exists': True, '$not': {'$size': 0}}}, {'issuetype.id': {'$exists': True, '$not': {'$size': 0}}}]}
mongo_projection = {'assignee': 1, 'summary': 1, 'description': 1, 'issuetype': 1,
                 'labels': 1, 'priority': 1, 'status': 1, 'projectname': 1}
issues = client['test']['issues'].find(filter=mongo_filter, projection=mongo_projection)
list_issues = list(issues)

# Keep only the desired values (ids) of the nested objects
for issue in list_issues:
    issue["type_id"] = issue["issuetype"]["id"]
    issue["priority_id"] = issue["priority"]["id"]
    issue["status_id"] = issue["status"]["id"]
    issue["type_name"] = issue["issuetype"]["name"]
    issue["priority_name"] = issue["priority"]["name"]
    issue["status_name"] = issue["status"]["name"]

# Convert data to dataframe and save to csv file
datapath = os.path.join(datafolder, "1_mongo.csv")
mongo_df = pd.DataFrame(list_issues, columns=['_id', 'projectname', 'assignee', 'summary', 'description', 'type_id',
                  'type_name', 'labels', 'priority_id', 'priority_name', 'status_id', 'status_name'])
mongo_df.to_csv(datapath, sep='\t', encoding='utf-8')