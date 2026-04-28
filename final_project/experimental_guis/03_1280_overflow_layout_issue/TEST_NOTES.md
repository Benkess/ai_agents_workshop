# Test Notes: 03 1280 Overflow Layout Issue

## Purpose
This fixture checks whether the GUI tester catches a page-level viewport fit problem in the personal website.

## Expected MCP Finding
The site should be flagged for not fitting cleanly in the target 1280x720 viewport, especially on the landing page.

## Ready Criteria After Fix
- The landing page fits within a 1280x720 viewport without unexpected top-level scroll or clipped primary content.
- Navigation, profile information, contact rows, main intro content, and footer remain visible and usable.
- Blog and resume pages remain complete and usable.
- The fix preserves the intended dark personal-site visual style.

## Notes
The agent-facing files are in `personal_webpage/`. Sample blog copy mentions layout and usability topics; that text is content, not the maintained issue note.
