# GUI Testing Report

- Generated: 2026-04-25T00:11:06Z
- Run directory: `C:\Users\benpk\Projects\computer-use-agent-tests\reports\run_20260424_201043`

## Summary Of Task

Test Avery Hart's static personal website at 1280x720, focusing on whether index.html fits without vertical scrolling or clipped content, and verify that the shared navigation and footer are present on index.html, blog.html, and resume.html, with blog/resume content readable.

## Results

On index.html at 1280x720, the page appears to fit entirely within the viewport. I observed no vertical scrollbar in the provided home-page screenshot. The sidebar was fully visible, including the name, role summary, and all four contact rows: Email, ORCID, GitHub, and LinkedIn. The main content was fully visible, including the welcome heading ('Thanks for visiting!') and all three bio paragraphs, with no clipping observed. The footer was visible at the bottom of the page. On blog.html, the shared top nav and footer were present and the blog content was readable; a scrollbar was visible inside the blog content area/container, which is acceptable per the instructions. On resume.html, the shared top nav and footer were present and the resume content was readable, with Experience, Education, Projects, and Interests visible.

## Important Findings

Index.html passes the stated 1280x720 layout goal based on the observed screenshot: no visible vertical scrollbar and no clipped content. Specifically, the home page showed the full sidebar, the full main bio content, and the footer all within the viewport. Blog and Resume pages also showed the expected nav and footer, and their content was readable.

## Suggestions

If this layout is critical, consider adding an automated visual regression check at 1280x720 to ensure the home page continues to fit with no scrollbar after future content changes.

## Other Notes

Navigation clicks triggered page loads successfully, though the automation tool reported transient 'execution context was destroyed' messages during navigation; these did not prevent verification because the destination pages loaded and were visible.

## Linked Notes

- [Note 001](notes/note_001.md)
- [Note 002](notes/note_002.md)
- [Note 003](notes/note_003.md)
