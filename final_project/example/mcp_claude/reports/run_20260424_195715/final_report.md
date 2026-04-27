# GUI Testing Report

- Generated: 2026-04-24T23:57:47Z
- Run directory: `C:\Users\benpk\Projects\computer-use-agent-tests\reports\run_20260424_195715`

## Summary Of Task

Test the Avery Hart personal website across index.html, blog.html, and resume.html for required landing-page content, shared navigation, blog and resume content, and overall dark-theme layout/readability at 1280x720.

## Results

I tested all three pages by using the shared top navigation. The landing page uses a dark charcoal background with light readable text, prominently displays 'Avery Hart', and includes a short role summary mentioning robotics and computer vision. It also includes welcome/bio text and selected interests in the main content area. The top navigation bar was visible on all tested pages and contained links for Avery Hart (home/index), Blog Posts, and Resume; clicking these links navigated to the expected pages. The blog page showed a clear 'Blog Posts' title, subtitle text, and several realistic post previews with titles and summaries in a readable dark-themed layout. The resume page showed clear sections for Experience, Education, and Projects with compact readable entries and bullets; layout was usable and consistent with the rest of the site.

## Important Findings

1. Landing page: the left-side contact/profile rows did not fully meet spec in the visible viewport. Email, ORCID, and GitHub were visible, but LinkedIn was not present as its own visible contact row in the sidebar list. A LinkedIn text link appears in the main paragraph instead, which does not match the specified contact/profile row requirement. 2. Landing page: main content is clipped/overflowing at 1280x720. The lower portion of the paragraph content is cut off by the footer/container boundary, reducing readability of important content in the target viewport. 3. Blog page: post previews are placed inside a fixed-height inner scroll area with its own scrollbar. Content remains readable, but this is a usability concern and makes the layout less natural than a normal page flow.

## Suggestions

Add a dedicated LinkedIn contact row to the landing-page profile/contact list to match the spec. Adjust landing-page spacing/container height so all key intro content is visible without clipping at 1280x720. Consider removing the inner scrolling panel on the blog page so posts flow naturally on the page.

## Other Notes

Design was otherwise coherent across all three pages: consistent dark theme, typography, spacing, top navigation, and footer styling. No broken navigation links were observed during testing.

## Linked Notes

- [Note 001](notes/note_001.md)
- [Note 002](notes/note_002.md)
- [Note 003](notes/note_003.md)
