# Final Project

## Video Presentation

- [Video Presentation](https://youtu.be/CR5Ww6mZbIk)

## Final Report

- [Report Document](/final_project/AIW_Final_Project_Report.md)
- [Assignment Instructions](/final_project/assignment.md)
- Report Figures: `final_project/report_figures`

## Experimental Artifacts

- **Experimental GUIs:** Located in `final_project/experimental_guis`. These are the Codex generated GUIs used in the evaluation.
- **MCP Test Trials:** Located in `final_project/example`. The artifacts from one experimental trial are saved for each coding agent used in the experiments (Claude Code and Codex). The artifacts include: 
    - a readme that notes the coding agent and experimental GUI name,
    - a coding agent chat log (documenting the conversation),
    - a reports directory with all the GUI tester sub agent run outputs,
    - and a fixed code directory with the updated code produced by the coding agent at the conclusion of the trial.

## Code

The source code for the GUI tester package is located in the [gui-tester gitrepo](https://github.com/Benkess/gui-tester.git).

It is also included as submodule at `/final_project/gui-tester`. To initialize the submodule run the following command after cloning this git repo:

```bash
git submodule update --init --recursive
```