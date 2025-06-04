# Survey directory description

This document describes the CSV files in the `/survey` directory

## feedback_data.csv

This file contains participant feedback.

This file contains participant feedback.

| Column name | Description                               |
| ----------- | ----------------------------------------- |
| exp_name    | Experiment name                           |
| task_type   | Type of task (training or evaluation set) |
| hashed_id   | Anonymized participant identifier         |
| feedback    | Feedback provided by the participant      |

## demographics_data.csv

This file contains demographic information about the participants.

| Column name     | Description                                                |
| --------------- | ---------------------------------------------------------- |
| exp_name        | Experiment name (internal identifier)                      |
| task_type       | Type of task (training or evaluation set)                  |
| hashed_id       | Anonymized participant identifier                          |
| age             | Age of the participant                                     |
| gender          | Gender of the participant                                  |
| race            | Race of the participant                                    |
| education_level | Education level of the participant                         |
| normal_vision   | Boolean indicating if the participant has normal vision    |
| color_blind     | Boolean indicating if the participant is color blind       |
| fluent_english  | Boolean indicating if the participant is fluent in English |

## withdraw_data.csv

This file contains information about participants who withdrew from the experiment.

| Column name      | Description                                    |
| ---------------- | ---------------------------------------------- |
| exp_name         | Experiment name                                |
| task_type        | Type of task (training or evaluation set)      |
| hashed_id        | Anonymized participant identifier              |
| withdraw         | Boolean indicating if the participant withdrew |
| withdraw_reason  | Reason for withdrawal                          |
| withdraw_comment | Comment on withdrawal                          |
