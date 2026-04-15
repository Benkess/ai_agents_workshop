# GUI Tester TODO

This file tracks follow-up work discovered while building and testing the first working version of the GUI tester. These are not all urgent, but they are important enough that we should not lose them.

## Output and Packaging

### Move report output out of the package source tree by default

Observation:
- The first successful run wrote reports into `final_project_gui_tester/reports/` inside the repo.
- This works for local development, but it clutters the source tree and requires manual cleanup.
- When this project is later packaged for use in another repo or editor toolchain, we should not depend on a pre-created reports directory inside the package.

Follow-up:
- Decide on a packaging-safe default output location.
- Keep `report_dir` override support so callers can still force reports into a known location.
- Preserve the returned `report_path` as the main handoff contract.

### Decide how to handle ignored output files

Observation:
- Repo-local reports are annoying if they are tracked or visible all the time.
- Git-ignoring them would be cleaner for development, but some coding-agent setups may treat ignored files differently or may not index/read them as expected.

Follow-up:
- Confirm whether target coding-agent environments can reliably access gitignored output files by path.
- Decide whether default output should live outside the repo instead of relying on `.gitignore`.

## Prompt and Behavior Tuning

### Stop telling the tester to reopen the starting page

Observation:
- The browser was already launched on the correct page by the wrapper.
- The runtime prompt also included the starting URL.
- In the first real run, the tester repeatedly called `navigate` back to the starting page even though it was already there.

Follow-up:
- Reword the runtime context so the tester understands it starts on the correct page.
- Consider not exposing the raw starting URL in the prompt unless needed for the task.

Status update:
- We changed the prompt so the tester starts from the already-open page and removed the model-facing URL reference.
- On the next rerun, the repeated `navigate` behavior stopped and the tester stayed on task.

### Remove or rethink the `screenshot` action in the testing `computer_use` tool

Observation:
- The harness already captures a screenshot before each step.
- During the first run, the tester still called the `computer_use` `screenshot` action several times.
- Those calls were likely used as an observation or evidence-gathering step, but they were redundant and not part of the intended flow.

Follow-up:
- Consider removing `screenshot` from the testing-specific tool.
- If the model still needs a way to request a screenshot explicitly, define a clearer harness-side behavior later.

### Reduce low-value note spam

Observation:
- In the first run, the tester logged notes almost every step.
- Many of those notes were repetitive setup statements like "Starting GUI test..." rather than meaningful findings.
- This made the notes directory noisy even though the final report was still useful.

Follow-up:
- Adjust prompt guidance so notes are saved for meaningful findings, blockers, or important state transitions rather than general thoughts.
- Consider making note-taking more clearly optional at each step.
- Consider adding an explicit "no note needed" path if the model keeps overusing the note tool.

Status update:
- We tightened the prompt and tool description so notes are for meaningful findings rather than narration.
- On the next rerun, note count dropped from 17 notes to 3 and the remaining notes were much higher quality.

### Clarify the intended use of notes versus final report content

Observation:
- The tester used notes for both real findings and lightweight progress narration.
- The final report was better curated than the note set, which suggests the agent can summarize well but does not yet know what deserves a note.

Follow-up:
- Clarify in the prompt that notes are evidence or findings, not a step-by-step diary.
- Consider defining note criteria such as: bug found, visual evidence worth saving, blocker hit, or important page-level conclusion.

## Tooling and Harness Behavior

### Review `tool_choice="required"` effects in the tester workflow

Observation:
- The custom agent requires at least one tool call each turn.
- In practice, the tester often emitted both `computer_use` and `gui_testing_report_tool` in the same step.
- This likely contributes to note spam and may bias the model toward unnecessary tool use.

Follow-up:
- Re-evaluate whether the tester should keep the same required tool-calling behavior as the base computer-use demo.
- If it stays required, tune prompts and tool descriptions to reduce unnecessary extra calls.

### Improve post-click behavior when navigation occurs

Observation:
- Several click actions were followed by tool errors like `Execution context was destroyed, most likely because of a navigation`.
- Despite those reported errors, the page often navigated successfully and the tester continued correctly.
- This appears to come from cursor overlay update logic after a click triggers navigation.

Follow-up:
- Make the testing `computer_use` tool more navigation-safe around post-click page evaluation.
- Reduce false-negative tool results that make logs noisier than the actual browser behavior.

Additional observation:
- This issue still appeared in the improved rerun even though the overall agent behavior was much better.
- In that rerun, navigation clicks to Blog and Landing worked, but the tool still returned `Execution context was destroyed` errors immediately after the click.
- This is now one of the clearest remaining harness/tool quality issues.

### Remove or rethink the `screenshot` action after prompt tuning is validated further

Observation:
- After prompt tuning, the tester behavior improved a lot, but it still used the `computer_use` `screenshot` action at the beginning of the run.
- The harness already provides a screenshot each step, so this action still appears redundant in the tester workflow.

Follow-up:
- If this pattern persists across more runs, remove `screenshot` from the testing-specific tool or redesign how explicit screenshot requests should work.

## Memory and Long-Horizon Robustness

### Prevent loss of important findings due to sliding-window context

Observation:
- The base agent uses sliding-window message trimming.
- Early findings can fall out of context during longer runs.
- Even though findings are saved to notes on disk, the live agent does not automatically re-read them before writing the final report.

Risk:
- Important earlier bugs may be omitted from the final report.
- Longer runs may become repetitive or cause the agent to revisit already-tested areas.

Follow-up:
- Watch for this issue in more runs before over-engineering a fix.
- If it becomes a real problem, consider a second-pass summarizer over notes and the draft report.

### Watch for looping behavior on longer GUI tests

Observation:
- If the agent forgets earlier progress, it could revisit pages or re-run checks it already completed.
- This did not fully derail the first run, but the repeated re-navigation to the starting page shows the shape of the problem.

Follow-up:
- Keep this on the backlog while testing more complex GUIs.
- Add stronger progress-tracking or summarization only if repeated loops become common.

## Integration Readiness

### Revisit the parent-agent interface after direct testing is stable

Observation:
- The direct CLI path is working and was the right first milestone.
- Packaging for use by coding agents in editors like VS Code, Copilot, or Claude Code will add new constraints around output paths, tool registration, and file visibility.

Follow-up:
- Keep direct testing as the priority until the tester behavior is more stable.
- Return to parent-agent packaging and integration only after note quality, report quality, and output placement are in a better state.
