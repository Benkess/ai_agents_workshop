# Test Notes: 01 Missing Icons

## Purpose
This fixture checks whether the GUI tester notices visibly broken icon rendering in an otherwise usable personal website.

## Expected MCP Finding
The landing page contact rows should be flagged because the media/contact icons render as square placeholder glyphs instead of recognizable markers.

## Ready Criteria After Fix
- The landing page contact rows for Email, ORCID, GitHub, and LinkedIn show recognizable icon markers or clear textual alternatives.
- The label text remains visible and readable.
- Top navigation still includes Blog Posts and Resume.
- Blog and resume pages remain complete and usable.
- No unrelated layout or navigation regressions are introduced.

## Notes
The agent-facing files are in `personal_webpage/`. The parent directory name documents the fixture for maintainers only.
