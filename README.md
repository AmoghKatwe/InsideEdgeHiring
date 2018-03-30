## Terminology
1. **Subject**: A field that is grouped on; analogous to SQL's "GROUP BY"  
clause.  
2. **Split**: A filter used to restrict a dataset; analogous to SQL's "WHERE"  
clause.  
    * **vs LHH**: "versus left-handed hitters"  
    * **vs RHH**: "versus right-handed hitters"  
    * **vs LHP**: "versus left-handed pitchers"  
    * **vs RHP**: "versus right-handed pitchers"  
3. **Stat**: A metric that is calculated from the aggregated data. There are  
four basic stats to be calculated that should be familiar to any baseball fan.
    * AVG
    * OBP
    * SLG
    * OPS

## Instructions
1. Create a GitLab account (if you don't already have one).
2. Clone this repository to your machine.
3. Install it using `pip install -r requirements.txt`
3. Modify run.py to perform the following steps when called via `python run.py`:
    1. Read in `./data/raw/pitchdata.csv`
    2. Perform grouping/aggregations of each combination from  
    `./data/reference/combinations.txt` to create tables/dataframes.
    3. Round the stat to a max of three decimal places.
    4. Only include subjects with PA >= 25.
    5. Combine each individual table/dataframe into a single one with the  
    following column headers:
        * SubjectId _(e.g. 108, 119, etc)_
        * Stat _(e.g. the name of the stat "AVG", "OBP", etc.)_
        * Split _(e.g. "vs LHP", "vs RHH", etc.)_
        * Subject _(e.g. "HitterId", "PitcherTeamId", etc.)_
        * Value _(e.g. the value of the Stat 0.350, 1.03, 0.5, etc)_
    6. Sort the table/dataframe on the first four columns (each in ascending  
    order).
    7. Save the csv to `./data/processed/output.csv`
4. Run the test suite by opening a command-line, cd in to the repo, and running  
the following command: ```pytest -v```
5. Upload to a new repository under your own GitLab/GitHub/BitBucket account.
6. Email the link to your repository to Andrew Pautz (pautz@inside-edge.com).

## Example 
Let's take the first combination to be processed from combinations.txt:
```
AVG,HitterId,vs RHP
```
... the equivalent SQL would look something like:

```SQL
SELECT 
  HitterId AS SubjectId,
  'AVG' AS Stat,
  'vs RHP' AS Split,
  'HitterId' AS Subject,
  ROUND(CAST(SUM(H) AS FLOAT)/SUM(AB), 3) AS Value
FROM ./data/raw/pitchdata.csv
WHERE PitcherSide = 'R'
GROUP BY HitterId
HAVING SUM(PA) >= 50
```

## Goals
Your submission will be scored on 5 aspects:
1. **Accuracy**: The output data must be 100% accurate.
2. **Readability**: The easier it is to understand, the better.
3. **Performant**: It should ideally take just 1-2 seconds to finish.
4. **Development Time**: Try to submit within a day.
5. **Installable**: Make it installable via `pip install -r requirements.txt`

## Additional Info
* Use any third party libraries you'd like so long as they're installed via  
the requirements.txt file. We use pandas heavily, but if you're more  
comfortable using numpy or something entirely else feel free to do so.
* You don't need to limit your modifications to run.py. You can add/edit any  
other file in the repo with the exception of anything in:
    * ./tests
    * ./data/raw
    * ./data/reference
* Code should generally be pep8 compliant. 
    * Documentation isn't required, but wouldn't be frowned upon. 
    * If you need to go a bit over 80 chars ... no problem.
