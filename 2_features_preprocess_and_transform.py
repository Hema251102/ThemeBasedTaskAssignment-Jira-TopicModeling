import os
import pandas as pd
import matplotlib.pyplot as plt
from properties import datafolder, min_assignees, min_issues_per_assignee, num_assignees, all_assignees

# Read data from Mongo DB
mongo_df = pd.read_csv(os.path.join(datafolder, "1_mongo.csv"), sep='\t', encoding='utf-8')
# Drop unnamed column
mongo_df.drop(mongo_df.columns[mongo_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
# Convert assigneeID from str to int
mongo_df['assignee_id'] = mongo_df['assignee'].rank(method='dense').astype(int)
# Rename columns to be compliant with the chosen data schema
mongo_df.rename(columns={'_id': 'id', 'summary': 'title', 'projectname': 'project_name'}, inplace=True)
# Rearrange columns for better presentation of the data
mongo_df = mongo_df[['id', 'title', 'description', 'project_name', 'status_name', 'priority_id', 'type_id', 'assignee_id', 'labels']]

# Remove issues with no titles, descriptions
mongo_filtered_df = mongo_df[(mongo_df.description.notna() & (mongo_df.title.notna()))]
# Keep only the projects that have at least -min_assignees- AND at least -min_issues_per_assignee- assigned per assignee
projects_enough_issues_per_assignee = []
mongo_projects = mongo_filtered_df.project_name.unique()
for project in mongo_projects:
	issues_single_project = mongo_filtered_df[mongo_filtered_df['project_name'] == project]
	# Removes lines (issues) of assignees that were assigned less than min_issues_per_assignee issues
	removedAssigneeIds = [assigneeId for assigneeId in issues_single_project['assignee_id'].unique() if (len(issues_single_project[issues_single_project.assignee_id == assigneeId]) < min_issues_per_assignee)]
	issues_single_project = issues_single_project[~issues_single_project['assignee_id'].isin(removedAssigneeIds)]
	if len(issues_single_project.assignee_id.unique()) >= min_assignees:
		projects_enough_issues_per_assignee.append(project)
		print('Number of Assignees in', project, 'project: ', len(issues_single_project.assignee_id.unique()))
print("There are ", len(projects_enough_issues_per_assignee), "projects with at least", min_assignees, " assignees who were assigned at least", min_issues_per_assignee, " issues.")

for project_name in projects_enough_issues_per_assignee:
	print(project_name)
	for n_assignees in (all_assignees if project_name == "FLINK" else [num_assignees]):
		# Create a dataframe which contains the issues of this project
		mongo_single_project = mongo_filtered_df[mongo_filtered_df['project_name'] == project_name]
		# Save the n_assignees with the most topics
		assignees = mongo_single_project['assignee_id'].value_counts().iloc[:n_assignees].index.tolist()
		# Keep only issues that have an id present in the assignees list
		mongo_single_project_num_assignees = mongo_single_project[mongo_single_project['assignee_id'].isin(assignees)]
		# undersample the data
		msk = mongo_single_project_num_assignees.groupby('assignee_id')['assignee_id'].transform('size') >= 80
		mongo_single_project_num_assignees = pd.concat((mongo_single_project_num_assignees[msk].groupby('assignee_id').sample(n=80), mongo_single_project_num_assignees[~msk]), ignore_index=True)
		# write data to file
		mongo_single_project_num_assignees.to_csv(os.path.join(datafolder, "2_" + project_name + "_" + str(n_assignees) + "_assignees" + ".csv"), sep='\t', encoding='utf-8')