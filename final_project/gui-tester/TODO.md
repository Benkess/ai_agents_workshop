# gui_tester TODO

## Near-Term

- Add cleaner entrypoint-level missing-key/config errors for CLI and MCP startup.
- Add a documented MCP startup env var for advanced config overrides such as `GUI_TESTER_CONFIG`.
- Add Copilot and Codex MCP setup examples once the packaged draft is validated outside the workshop repo.
- Make dotenv not a dependancy for MCP use.
- Clean readme to be user facing.
- Add better setup instructions.
- Correct Claude Code setup.

## Packaging Follow-Up

- Re-test the package from a clean second repo or temp workspace after `pip install -e .`.
- Decide whether the bundled `comp_use` subtree should stay broad or be narrowed later.
- Split this directory into its own `gui-tester` git repo once the packaged draft is stable.
- Publish on PyPi.

## Behavior/UX Follow-Up

- Improve navigation-click logging so successful navigation clicks do not look like tool failures. Sometimes clicks return unsuccessful when they worked due to the overlay failing.
- Revisit note-taking and long-run memory if longer GUI tests start dropping earlier findings due to sliding context window.
