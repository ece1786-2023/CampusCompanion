# CampusCompanion

## Running the System

1. ChromaDB is running on docker, use docker compose
`docker compose up &`

2. To run the main system run:
`python main.py`

3. To run the UI
`cd UI`
`npm run dev`

## Testing
Test code in `test`:

* `generator.py` is for Synthetic data generation. It generate student's information and save into `student_info.json`.
* `stuModel.py` provides `StuModel` to act as a Student to interact with Advior and use Scoring Evaluator to evaluate the result.
