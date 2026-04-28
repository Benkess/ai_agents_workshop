# Test Notes: 02 Missing Blog Nav

## Purpose
This fixture checks whether the GUI tester notices that a real page exists but is not reachable through the expected visible navigation.

## Expected MCP Finding
The Blog Posts navigation item should be reported as missing from the top navigation, even though `blog.html` exists and loads directly.

## Ready Criteria After Fix
- The top navigation includes the person/site name, Blog Posts, and Resume.
- Blog Posts links to `blog.html` from each page.
- The blog page remains visually complete and usable.
- The resume link and home/person link still work.
- No unrelated visual regressions are introduced.

## Notes
The agent-facing files are in `personal_webpage/`. The parent directory name documents the fixture for maintainers only.
