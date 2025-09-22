"""Ankelodon Agent Adapter for the Hugging Face Agents Course evaluator.

This module exposes a simple Gradio-powered wrapper around the
`ankelodon_multiagent_system` project. It follows the same high-level flow
as the official GAIA template provided in the course materials: fetch
evaluation questions from the GAIA API, run your agent to produce
responses, and submit those responses back to the leaderboard.

The key differences between this adapter and the GAIA template are:

  * It imports and uses your multi‑agent system defined in the `src`
    package (see `src/agent.py`) via the `build_workflow` function. This
    function returns a `langgraph` state machine capable of planning,
    reasoning and executing tools. The adapter calls into this workflow
    with a properly initialised `AgentState` and extracts the final
    answer from the resulting state.
  * It automatically downloads any file attachments associated with a
    task (via the `/files/{task_id}` endpoint exposed by the evaluation
    server) and saves them into a temporary directory. The local file
    paths are passed into the agent through the `files` field of the
    state. Your existing file handling logic (e.g. `preprocess_files`
    in `src/tools/tools.py`) will detect the file type and suggest
    appropriate tools.
  * It strips any leading ``Final answer:`` prefix from the agent's
    response. The evaluation server performs an exact string match
    against the ground truth answer【842261069842380†L108-L112】, so it is
    important that the returned text contains only the answer and
    nothing else.

Before running this script yourself, make sure all dependencies in
`requirements.txt` are installed. To use the Gradio interface locally,
run `python ankelodon_adapter.py` from the project root. When deploying
as a Hugging Face Space for leaderboard submission, ensure the
`SPACE_ID` environment variable is set by the platform; it is used to
construct a link back to your code for verification.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, List, Dict, Any

import requests
import gradio as gr
import pandas as pd

try:
    # Import the multi‑agent system components. When running as a script
    # within the project root, Python's module search path should
    # already include the `src` directory. If you get import errors,
    # ensure that the working directory is the repository root or
    # append `src` to `sys.path` manually before these imports.
    from agent import build_workflow
    from config import config as WORKFLOW_CONFIG
    from state import AgentState
except Exception as import_err:
    raise RuntimeError(
        "Failed to import the Ankelodon multi-agent system. "
        "Make sure you are running this script from the repository root "
        "and that the project has been installed correctly."
    ) from import_err

DEFAULT_API_URL: str = "https://agents-course-unit4-scoring.hf.space"


class AnkelodonAgent:
    """Simple callable wrapper around the Ankelodon multi‑agent system.

    Instances of this class can be called directly with a natural
    language question and an optional task identifier. Under the hood it
    builds a `langgraph` workflow using ``build_workflow()``, prepares
    an initial state, fetches any file attachments associated with
    the task, and invokes the workflow to compute a final answer.
    """

    def __init__(self) -> None:
        # Initialise the workflow once per agent. Subsequent calls reuse
        # the compiled state machine, which is more efficient than
        # rebuilding it on every question.
        self.workflow = build_workflow()

    def _download_attachment(self, task_id: str) -> List[str]:
        """Download a file attachment for the given task ID.

        The evaluation API exposes a ``/files/{task_id}`` endpoint【842261069842380†L95-L107】.
        This helper downloads the content, infers a file extension
        from the HTTP ``Content-Type`` header and writes the bytes to a
        temporary file. It returns a list of file paths (zero or one
        element) to be included in the agent state.
        """
        files: List[str] = []
        url = f"{DEFAULT_API_URL}/files/{task_id}"
        try:
            resp = requests.get(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200 and resp.content:
                # Map common MIME substrings to file extensions. The
                # multi‑agent system's file handling tools use the
                # extension to determine how to process the file.
                ctype = resp.headers.get("content-type", "").lower()
                ext_map = {
                    "excel": ".xlsx",
                    "sheet": ".xlsx",
                    "csv": ".csv",
                    "python": ".py",
                    "audio": ".mp3",
                    "image": ".jpg",
                }
                extension = ""
                for key, val in ext_map.items():
                    if key in ctype:
                        extension = val
                        break
                tmp_dir = tempfile.mkdtemp(prefix="ankelodon_task_")
                filename = f"attachment{extension}"
                path = os.path.join(tmp_dir, filename)
                with open(path, "wb") as fh:
                    fh.write(resp.content)
                files.append(path)
        except Exception as e:
            # Log the error to console but don't fail the entire task.
            print(f"[WARNING] Failed to fetch attachment for task {task_id}: {e}")
        return files

    def __call__(self, question: str, task_id: Optional[str] = None) -> str:
        """Run the multi‑agent system to answer a question.

        Parameters
        ----------
        question: str
            The natural language query to answer.
        task_id: Optional[str]
            If provided, the ID used to fetch any associated file
            attachment from the evaluation API. Attachments are stored
            locally and passed into the agent via the ``files`` field.

        Returns
        -------
        str
            The final answer produced by the agent, with any "final
            answer" prefix removed. If no answer is produced the empty
            string is returned.
        """
        # Build the initial agent state. The AgentState type defines
        # numerous fields, many of which the workflow populates
        # internally. We set only the essentials here. Unrecognised
        # keys are ignored by the underlying state machine.
        state: Dict[str, Any] = {
            "query": question,
            "final_answer": "",
            "plan": None,
            "complexity_assessment": None,
            "current_step": 0,
            "reasoning_done": False,
            "messages": [],
            "files": [],
            "file_contents": {},
            "critique_feedback": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "execution_report": None,
            "previous_tool_results": {},
        }

        # If a task ID is provided, attempt to download its attachment.
        if task_id:
            attachment_paths = self._download_attachment(task_id)
            if attachment_paths:
                state["files"] = attachment_paths

        # Invoke the workflow. The `config` parameter defines runtime
        # options such as recursion limits and thread identifiers. It is
        # imported from `src.config`.
        try:
            result_state = self.workflow.invoke(state, config=WORKFLOW_CONFIG)
        except Exception as e:
            print(f"[ERROR] Failed to run workflow: {e}")
            return ""

        # Extract the final answer. Depending on the branch taken,
        # either the ``final_answer`` key or a generic ``answer`` key may
        # be present. Use whichever exists. Some nodes may prepend
        # "final answer:"; remove it for exact match scoring【842261069842380†L108-L112】.
        answer = ""
        if isinstance(result_state, dict):
            answer = result_state.get("final_answer") or result_state.get("answer") or ""
        if answer:
            answer = answer.replace("Final answer:", "").replace("final answer:", "").strip()
        return answer


def run_and_submit_all(profile: Optional[gr.OAuthProfile]) -> tuple[str, pd.DataFrame | None]:
    """Fetch all questions, run the agent, and submit the answers.

    This function replicates the behaviour of the GAIA template's
    ``run_and_submit_all`` function【566837548679297†L247-L306】 but uses the
    ``AnkelodonAgent`` class defined above. It is bound to a Gradio
    button in the UI. On success it returns a status message and a
    DataFrame of results; on failure it returns an error message and
    ``None`` or an empty DataFrame.
    """
    # Require the user to be logged in so we can report the username.
    if not profile:
        return "Please Login to Hugging Face with the button.", None
    username = getattr(profile, "username", "").strip()

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # Instantiate the agent once.
    try:
        agent = AnkelodonAgent()
        print("Ankelodon agent initialised successfully")
    except Exception as e:
        err_msg = f"Error initialising agent: {e}"
        print(err_msg)
        return err_msg, None

    # Fetch questions from the evaluation API.【566837548679297†L247-L268】
    try:
        print(f"Fetching questions from: {questions_url}")
        resp = requests.get(questions_url, timeout=15)
        resp.raise_for_status()
        questions_data = resp.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        err_msg = f"Error fetching questions: {e}"
        print(err_msg)
        return err_msg, None

    # Run the agent on each question.
    results_log: List[Dict[str, Any]] = []
    answers_payload: List[Dict[str, str]] = []
    print(f"Running agent on {len(questions_data)} questions…")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            answer = agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": answer,
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}",
            })

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # Prepare submission payload. The leaderboard displays a link to your
    # code; this is constructed from the SPACE_ID environment variable.
    space_id = os.getenv("SPACE_ID", "")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else ""
    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload,
    }

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        submission_resp = requests.post(submit_url, json=submission_data, timeout=60)
        submission_resp.raise_for_status()
        result_data = submission_resp.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        err_msg = f"Submission Failed: {e}"
        print(err_msg)
        return err_msg, pd.DataFrame(results_log)


# Build the Gradio interface. This interface resembles the official
# GAIA template【566837548679297†L372-L401】 but runs your Ankelodon agent.
with gr.Blocks() as demo:
    gr.Markdown("# Ankelodon Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions**
        
        1. Clone this repository or duplicate the associated Hugging Face Space.
        2. Log in to your Hugging Face account using the button below. Your HF
           username is used to attribute your submission on the leaderboard.
        3. Click **Run Evaluation & Submit All Answers** to fetch the questions,
           run the Ankelodon agent on each one, submit your answers, and display
           the resulting score and answers.
        
        ---
        This template is intentionally lightweight. Feel free to customise it –
        add caching, parallel execution or additional logging as you see fit.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    # When running locally, print some information about the environment.
    print("\n" + "-" * 30 + " Ankelodon Adapter Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")
    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")
    print("-" * (60 + len(" Ankelodon Adapter Starting ")) + "\n")
    # Launch the Gradio app.
    demo.launch(debug=True, share=False)