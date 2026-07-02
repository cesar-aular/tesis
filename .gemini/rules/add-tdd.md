# Agent-Driven Development (ADD) & TDD Mandates

To maintain control and prevent hallucinations or technical debt during our sessions, you MUST STRICTLY comply with the following behavioral rules:

1. **Strict TDD Protocol (Red-Green-Refactor):**
   - It is FORBIDDEN to write or modify an ETL script or ML model directly without first creating its respective test in the `tests/` folder.
   - You will first show the user what you are going to test. Then you will create the test (e.g., `tests/test_cruce.py`), run `pytest`, confirm it fails (Red), and only then implement the source code to make it pass (Green).

2. **Short-Term Memory Protocol (Checkpoints):**
   - Do not finish a large task without writing or updating a `docs/tasks.md` file detailing exactly what we did, what is working, and what the next logical step is.
   - If the user says "continuemos" (let's continue) in a new session, your first mandatory action is to use `view_file` to read `docs/tasks.md` before proposing new code.

3. **Active Use of Knowledge Items (KIs):**
   - If we discover a complex solution (e.g., fixing the `numeric_only` bug in pandas or how to do Fuzzy Matching), create a Markdown file documenting it in `.gemini/knowledge/`.
   - Before writing new code, mentally review if there is a relevant KI in that folder so you don't reinvent the wheel.
