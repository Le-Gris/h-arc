# Data directory description

This document describes the CSV files in the `/data` directory.

## data.csv

This file contains all collected task data for complete and incomplete participant data, respectively. Participant data was deemed incomplete if the data file had an insufficient number of attempted tasks (5 or 10 depending on the experiment).

Each row represents a single, numbered action taken by a unique participant on a given task and attempt with all relevant experiment, participant, task and action information.

| Column name            | Description                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| exp_name               | Experiment name (internal identifier)                                                     |
| task_type              | Type of task (training or evaluation set)                                                 |
| hashed_id              | Anonymized participant identifier                                                         |
| joint_id_task          | Combined identifier for participant and task                                              |
| task_name              | Name of the task                                                                          |
| task_number            | Number of the task (i.e., 3 is the third task completed)                                  |
| is_tutorial            | Boolean indicating if the task is a tutorial                                              |
| time                   | Timestamp of the action                                                                   |
| attempt_number         | Number of the attempt                                                                     |
| action_id              | Number of the action taken                                                                |
| solved                 | Boolean indicating if the task was solved at this action                                  |
| done                   | Boolean indicating if the attempt was completed                                           |
| test_input_grid        | Input grid for the task                                                                   |
| test_input_size_x      | X-dimension of the input grid                                                             |
| test_input_size_y      | Y-dimension of the input grid                                                             |
| test_output_grid       | Output grid for the task in string format                                                 |
| test_output_size_x     | X-dimension of the output grid                                                            |
| test_output_size_y     | Y-dimension of the output grid                                                            |
| action                 | Action taken by the participant                                                           |
| action_x               | X-coordinate of the action                                                                |
| action_y               | Y-coordinate of the action                                                                |
| select_loc             | Selected location                                                                         |
| selected_data          | Data selected by the participant                                                          |
| selected_symbol        | Symbol selected by the participant (i.e., color in the experiment interface)              |
| selected_tool          | Tool selected by the participant                                                          |
| copy_paste_data        | Data used in copy-paste actions                                                           |
| first_written_solution | First natural language solution written by the participant                                |
| last_written_solution  | Last natural language solution written by the participant                                 |
| withdraw               | Boolean indicating if the participant withdrew                                            |
| withdraw_reason        | Reason for withdrawal                                                                     |
| withdraw_comment       | Comment on withdrawal                                                                     |
| age                    | Age of the participant                                                                    |
| gender                 | Gender of the participant                                                                 |
| race                   | Race of the participant                                                                   |
| education_level        | Education level of the participant                                                        |
| household_income       | Household income of the participant                                                       |
| normal_vision          | Boolean indicating if the participant has normal vision                                   |
| color_blind            | Boolean indicating if the participant is color blind                                      |
| fluent_english         | Boolean indicating if the participant is fluent in English                                |
| complete               | Boolean indicating if the data is from a participant that completed the experiment or not |

## summary_data.csv

This file contains summary data for complete and incomplete participant data, respectively. Each row represents a summary of an attempt by a unique participant at a given task.

| Column name            | Description                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| exp_name               | Experiment name (internal identifier)                                                     |
| task_type              | Type of task (training or evaluation set)                                                 |
| hashed_id              | Anonymized participant identifier                                                         |
| joint_id_task          | Combined identifier for participant and task                                              |
| task_name              | Name of the task                                                                          |
| task_number            | Number of the task (i.e., 3 is the third completed task)                                  |
| attempt_number         | Number of the attempt                                                                     |
| num_actions            | Number of actions taken until submission                                                  |
| solved                 | Boolean indicating if the task was solved                                                 |
| test_output_grid       | Output grid for the task                                                                  |
| first_written_solution | First solution written by the participant                                                 |
| last_written_solution  | Last solution written by the participant                                                  |
| complete               | Boolean indicating if the data is from a participant that completed the experiment or not |

## incorrect_submissions.csv

These files contain error information for complete and incomplete participant data, respectively.

| Column name      | Description                               |
| ---------------- | ----------------------------------------- |
| task_name        | Name of the task                          |
| task_type        | Type of task (training or evaluation set) |
| test_output_grid | Output grid for the task                  |
| count            | Number of occurrences of this error       |
