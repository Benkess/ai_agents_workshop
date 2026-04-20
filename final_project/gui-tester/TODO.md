# gui_tester TODO

## Near-Term

- Add cleaner entrypoint-level missing-key/config errors for CLI and MCP startup.
- Add a documented MCP startup env var for advanced config overrides such as `GUI_TESTER_CONFIG`.
- Add Copilot and Codex MCP setup examples once the packaged draft is validated outside the workshop repo.

## Packaging Follow-Up

- Re-test the package from a clean second repo or temp workspace after `pip install -e .`.
- Decide whether the bundled `comp_use` subtree should stay broad or be narrowed later.
- Split this directory into its own `gui-tester` git repo once the packaged draft is stable.

## Behavior/UX Follow-Up

- Improve navigation-click logging so successful navigation clicks do not look like tool failures.
- Revisit note-taking and long-run memory if longer GUI tests start dropping earlier findings.
- Decide later whether the MCP tool should return a short structured summary in addition to `report_path`.
