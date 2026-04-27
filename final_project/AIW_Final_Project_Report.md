# GUI Tester Subagent

**Ben Kessler**  
**CS 6501 Workshop on Building AI Agents**  
**Final Project Report — First Draft**

## Motivation — Why I Built It

Modern coding agents are increasingly capable of generating working front-end code from a short natural-language specification. They can create HTML, CSS, JavaScript, and even multi-page web applications with very little human guidance. However, in practice, a major weakness remains: coding agents often struggle to evaluate the visual correctness and usability of the graphical interfaces they create. A program can pass normal code checks, contain no obvious syntax errors, and still be visually broken when viewed in a browser.

This project was motivated by that gap between code generation and visual validation. In my own experiments, I found that coding agents could produce web pages that were technically functional but still had user-facing issues. Examples included content that did not fit cleanly inside the target viewport, sections of text being cut off, navigation links missing from some pages, and icons or contact links not appearing where the specification required them. These are the kinds of bugs that are easy for a human to notice by looking at the interface, but they are difficult for a code-only agent to reliably catch without being able to inspect and reason about the rendered GUI.

The core problem is what I call the visibility gap. Coding agents can read the source files, edit them, and run command-line checks, but many GUI problems are only obvious once the application is actually opened. When the agent cannot independently inspect the interface, the human user has to become part of the testing loop. The user must open the site, notice visual problems, take screenshots, describe the issue, and ask the agent to revise the implementation. This creates iteration friction: even if the agent can make the fix once the problem is described, the process still depends on repeated human inspection and feedback.

The goal of this project was to reduce that friction by giving coding agents a way to test the GUIs they build. Instead of requiring the user to manually inspect every generated interface, the coding agent should be able to call a separate GUI testing subagent. That subagent should open the application, interact with it, save notes and screenshots, and return a structured report describing whether the interface matches the specification. The coding agent can then use that report to make fixes and run the tester again.

In short, I built the GUI Tester Subagent to make GUI development more autonomous. The larger goal was not just to detect a single kind of bug, but to create a reusable testing layer that coding agents such as Codex or Claude Code could call when they need visual feedback.

## Methods — How I Built It

The system is organized into three main layers: a computer-use agent, a GUI testing agent, and a Model Context Protocol (MCP) integration layer. Each layer adds a different piece of functionality.

### Computer-Use Agent

The first layer is the computer-use agent. This is the base agent responsible for interacting with the environment in a human-like way. It can observe the screen, reason about what it sees, and execute standard actions such as clicking, typing, and dragging.

The agent uses a ReAct-like loop. At each step, the agent observes the current screen image, produces reasoning about what it should do next, selects an action, and executes that action. This repeats until the task is complete or the agent decides it has enough information to report its findings.

The computer-use agent was implemented using a LangGraph/LangChain-style orchestration framework. For model backends, I experimented with GPT-5.4 and Qwen3-VL:4B. The agent maintains context using a sliding window over recent interaction history, including recent observations, tool calls, and tool responses. In addition to this dynamic context, it keeps a static base context containing the system prompt and the user’s original task instructions.

For environment control, the agent supports core computer-use commands such as click, drag, and type. Depending on the setting, these commands can be executed through PyAutoGUI or through Playwright for browser-based testing. This layer provides the general ability to operate a GUI, but it does not by itself define what should be recorded during a test or how the result should be returned to a coding agent.

![Figure 1](/final_project/report_figures/computer_use_agent_architecture.jpg)  
Figure 1\. Computer-use agent architecture diagram

### GUI Testing Agent

The second layer is the GUI testing agent. This agent inherits the basic observation, reasoning, and action capabilities of the computer-use agent, but adds tools specifically designed for GUI evaluation.

The most important addition is a GUI testing tool with two main capabilities: note-taking and report generation. During a test, the agent can save observations as notes. These notes can optionally include screenshots, so that each finding is tied to visual evidence. At the end of a test, the agent can consolidate its notes into a final structured report.

This design is important because the goal is not only to let the agent look at a GUI, but also to create artifacts that a coding agent can use later. A screenshot by itself is useful to a human, but a coding agent also needs a written explanation of what is wrong and why it matters. The GUI testing agent therefore produces both visual and textual outputs. For example, it can record that the page mostly matches the requested dark theme, but that a contact link is missing from the visible sidebar or that a paragraph is clipped at the bottom of the viewport.

The GUI testing agent is also given the test instructions and the expected GUI description. This allows it to compare the rendered interface against the specification rather than only making general comments about visual quality. For instance, if the specification says the site should include Email, ORCID, GitHub, and LinkedIn contact links, the tester can explicitly check whether all of those links are present and visible.

![Figure 2](/final_project/report_figures/GUI_testing_agent_details.jpg)  
Figure 2\. GUI testing agent details

### MCP Integration

The third layer is the Model Context Protocol integration. The MCP layer wraps the GUI testing agent as a tool that can be called by a coding agent.

The MCP wrapper exposes a tool that takes four main inputs from the coding agent:

1. the URL or local path of the GUI to test,  
2. a description of the intended GUI,  
3. test instructions describing what the GUI tester should verify, and  
4. a report directory where output artifacts should be saved.

After running the GUI testing agent, the MCP tool returns the path to the final test report and the associated artifacts, including screenshots and saved notes. This makes the GUI tester easy to use from agentic coding environments. In my evaluation, I tested this integration with both Claude Code and Codex.

The key design decision was to make the GUI tester a separate subagent rather than building the entire workflow into the coding agent itself. This separation allows the coding agent to focus on implementation while delegating visual inspection to a specialized tester. It also makes the tester reusable across different coding agents. As long as a coding agent can call the MCP tool and read the resulting report, it can use the GUI tester to guide its own debugging process.

![Figure 3](/final_project/report_figures/MCP_integration_flow.jpg)  
Figure 3\. MCP integration flow 

## Evaluation — How Well It Worked

I evaluated the system through a small set of GUI testing experiments. The goal was to see whether the GUI Tester Subagent could identify real visual and navigational issues that coding agents failed to catch on their own, and whether those agents could then use the MCP output to fix the interface.

### Test GUIs

For the test cases, I used four buggy GUIs. These GUIs were generated by Codex from initial specifications for working interfaces. Some of the bugs were real errors introduced by the coding agent during the initial generation process. This made the evaluation realistic: instead of hand-crafting artificial bugs, I tested on the kinds of mistakes that can naturally appear in agent-generated front-end code.

The main example was a personal website template for a fictional person named Avery Hart. The intended site included a landing page, blog page, and resume page. It used a dark theme and included profile information, contact links, navigation, and page-specific content.

When viewed in the browser, the initial implementation looked mostly plausible at first glance. However, it contained several issues that were not reliably caught by code-level inspection. Across the experimental GUIs, the recurring issue types included:

- content not fitting cleanly inside the target viewport,  
- sections or paragraphs being cut off,  
- missing navigation links on some pages,  
- missing or hidden icons and contact links, and  
- layouts that technically rendered but did not satisfy the requested visual specification.

In the personal website example, the GUI tester found that the main content was clipped near the bottom of the viewport at the target 1280×720 size. It also found that the LinkedIn contact link was not visible in the left contact rows, even though the specification expected contact links such as Email, ORCID, GitHub, and LinkedIn to be available.

![Figure 4](/final_project/report_figures/00_valid_personal_webpage_index.png)  
Figure 4\. Intended personal website interface

![Figure 5](/final_project/report_figures/annotated_incorrect_guis.png)  
Figure 5\. Red-box annotated examples of visually incorrect GUIs

![Figure 6](/final_project/report_figures/04_annotated.png)  
Figure 6\. Red-box annotated example showing the cut-off contact links and clipped bio.

### Individual Testing of the GUI Tester

I first tested the GUI testing agent by itself, outside of the full coding-agent loop. In this setting, the tester was given the GUI, the expected description, and instructions for what to inspect. It then opened the page, navigated through the interface, saved observations, and generated a report.

This individual testing was meant to verify that the GUI testing agent’s own tool use was reliable. In particular, I wanted to confirm that it could capture screenshots, save notes, and identify visible problems in the interface. In the personal website case, the tester generated a note for the landing page stating that the main content text was clipped near the bottom of the viewport, suggesting overflow or content cut-off at 1280×720. It also noted that LinkedIn was not visible in the left contact rows.

The output artifacts were useful because they connected each finding to evidence. A saved note could include both the textual observation and the corresponding screenshot. This makes the report easier for a human to audit and easier for a coding agent to use when deciding what to fix.

![Figure 7](/final_project/report_figures/note_001.png)  
Figure 7\. Example saved note with screenshot from the GUI tester.

### MCP Testing With Coding Agents

After testing the GUI tester independently, I evaluated it as an MCP tool used by coding agents. I tested the MCP integration with both Claude Code and Codex.

The baseline result was that neither coding agent reliably identified the GUI bugs on its own. Without using the GUI tester, the agents could inspect the source code and reason about what the page was supposed to do, but they did not independently notice the visual issues. This matched the motivation for the project: the bugs were not primarily syntax or logic errors, but rendered-interface problems.

When given access to the MCP-based GUI tester, however, the coding agents were able to use the tool to catch the issues. The workflow was:

1. the coding agent generated or opened the GUI implementation,  
2. the coding agent called the GUI tester through the MCP tool,  
3. the GUI tester inspected the interface and produced a report,  
4. the coding agent read the report and patched the source code,  
5. the coding agent ran the GUI tester again,  
6. the loop repeated until the tester reported that the issues were resolved.

This iterative process worked especially well for the personal website example. The tester first reported the missing/hidden LinkedIn contact row and the clipped landing-page content. The coding agent then modified the layout and spacing. On the first fix attempt, the tester could still identify remaining issues. The coding agent then made additional changes and tested again. Eventually, the GUI tester reported that the implementation met the specification, and the final rendered page no longer had the original cut-off content or missing contact link.

![Figure 8](/final_project/report_figures/final_report_top_half.png)  
Figure 8\. GUI testing report summary, results, and important findings.

![Figure 9](/final_project/report_figures/final_report_bottom_half.png)  
Figure 9\. GUI testing report suggestions, linked notes, and follow-up recommendations.

![Figure 10](/final_project/report_figures/fixed_gui.png)  
Figure 10\. Fixed personal website template after iterative MCP-guided repair.

### Evidence and Limitations

I tested the system on four buggy GUIs and observed whether the GUI tester could find the intended issues and whether coding agents could use the tester to repair the GUIs. The experiments provide useful evidence that the approach works in the intended setting. The important result is not simply that the GUI tester could identify a bug, but that it could become part of an autonomous development loop. The coding agents failed to catch the issues independently, then succeeded once they used the GUI tester through MCP. This supports the main claim of the project: a specialized GUI testing subagent can give coding agents the visual feedback they need to fix problems that would otherwise require human inspection.

There are several limitations. First, the evaluation set was small. Four GUI examples are enough to demonstrate the prototype, but not enough to measure general reliability. Second, the current tester depends on the visual reasoning ability of the underlying model. If the model misses a subtle visual issue, the tester may fail to report it. Third, the current workflow is strongest for relatively simple front-end GUIs. More complex applications with login flows, dynamic state, animations, or accessibility requirements would need more detailed test instructions and possibly additional tools. Finally, the system currently focuses on qualitative findings rather than standardized scoring, so future work would need a more formal evaluation method.

## Conclusions — What I Learned

This project showed that GUI testing can be productively separated from code generation. Coding agents are good at writing and editing code, but they still need reliable feedback about what the generated interface actually looks like. By wrapping a computer-use GUI testing agent as an MCP tool, the coding agent can delegate visual inspection to a specialized subagent and use the resulting report to guide further edits.

The main thing I learned is that even a small amount of visual feedback can significantly improve an agentic coding workflow. The bugs in my test GUIs were not especially complex, but they were exactly the kinds of issues that interrupt real GUI development: a link is missing, a paragraph is clipped, a viewport assumption is wrong, or a page does not match the requested navigation structure. These issues are easy for humans to notice, but they create friction when the coding agent has to wait for the user to point them out. The GUI Tester Subagent reduces that friction by letting the agent inspect its own work.

I also learned that artifact design matters. The tester’s value does not come only from observing screenshots. It comes from saving structured notes, screenshots, and final reports that can be consumed by both humans and agents. The report acts as a bridge between visual evidence and code changes.

Overall, the project suggests that MCP-based subagents are a practical way to extend coding agents with specialized abilities. In this case, the MCP let Claude Code and Codex use a GUI tester to catch and fix issues they otherwise missed. The final result was a workflow where the coding agent could independently test a GUI, apply patches, and repeat the process until the interface satisfied the specification.

In future work, I would extend the evaluation to a larger benchmark of generated GUIs, add more systematic pass/fail criteria, and compare the GUI tester against human ratings or traditional automated UI checks. I would also like to add stronger accessibility checks, responsive layout testing across multiple screen sizes, and richer reporting formats. However, even in its current prototype form, the GUI Tester Subagent demonstrates the core idea: coding agents can become more autonomous when they are given tools to visually validate the interfaces they create.  
