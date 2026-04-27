# Test Notes: 04 Text Clipped Panels

## Purpose
This fixture checks whether the GUI tester catches content panels that visually clip important text in the personal website.

## Expected MCP Finding
The tester should report that text content is cut off or cramped inside one or more rendered panels, making the page feel incomplete or hard to read.

## Ready Criteria After Fix
- Landing, blog, and resume content panels display their important text without visual clipping.
- Panel scrolling, if used, is clear and usable.
- The site still fits cleanly in the target viewport.
- Navigation and page structure remain unchanged.

## Notes
The agent-facing files are in `personal_webpage/`. Sample blog copy mentions layout and usability topics; that text is content, not the maintained issue note.
