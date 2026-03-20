# Scope: Plan Page Improvements

**Due date:** Tuesday  
**Parent:** Plan Page Improvements — all Plan page UI and behaviour changes  
**Status:** Draft  

*Estimates below are based on review of the Plan page, Code page, and Mermaid components.*

---

## 1. Overview

Scope and effort for Plan page improvements. Sub-tasks touch the Plan page (`task/[taskId]/plan/page.tsx`), the Code page task cards (`task/[taskId]/code/page.tsx`), and the shared Mermaid component. Effort is informed by current implementation.

---

## 2. Sub-tasks & Effort (Code-Informed)

| # | Sub-task | Effort | Code / UX context |
|---|----------|--------|--------------------|
| 1 | **Mermaid syntax updation for padding & readability (refer to Jayesh screenshot)** | **Medium** | Mermaid is used in `plan/page.tsx` (slice Architecture block via `<MermaidDiagram chart={item.architecture} />`). Shared component: `components/chat/MermaidDiagram.tsx` (preprocessing, render, config); styles in `globals.css` (`.mermaid-diagram`). Work: adjust MermaidDiagram wrapper/CSS and/or `mermaid.initialize` options for padding and readability; align with reference screenshot. |
| 2 | **Replace test diff, test code and code gen with single Code Gen button** | **Medium** | Code page `MockTaskCard` has three tabs: "Test Diff", "Test Code", "Code Gen" (lines ~2043–2079). "Generate Code" button is shown when on Code Gen tab. Scope: remove Test Diff and Test Code tabs; keep only one entry point (Code Gen) and its content/streaming. Update state/tab logic and any copy that references the removed tabs. |
| 3 | **Remove the file names in each test** | **Small** | Code page: task card header shows `task.file` (e.g. `task.title` + `task.file`); Test Diff tab has a header with `task.file` in a bar above the diff. Remove these file-name displays from the card and from inside the test/code views. |
| 4 | **Remove description, implementation steps, context handoff, specs to generate** | **Medium** | Plan page slice accordion (`plan/page.tsx`) currently has four sections: Description (lines ~377–381), Implementation Steps (~385–397), Context Handoff (~408–424), Specs to Generate (~456–506). Remove these four blocks from the accordion content. May need to keep data in types/API for backward compatibility or strip only from UI. |
| 5 | **Add slices to the Plan page** | **Medium** | Plan page today shows slices only in a single accordion list. Code page has a left sidebar "Slices" with timeline (slice list + active/completed state). Scope: add a Slices list or sidebar to the Plan page so users can see and optionally navigate slices (e.g. list of slice titles with clear numbering/order), consistent with Code page or as per design. |

---

## 3. File / Area Summary

| Area | Files / locations |
|------|-------------------|
| Plan page content & structure | `app/(main)/task/[taskId]/plan/page.tsx` |
| Code page task cards & tabs | `app/(main)/task/[taskId]/code/page.tsx` (MockTaskCard, tabs, file display) |
| Mermaid styling & behaviour | `components/chat/MermaidDiagram.tsx`, `app/globals.css` (`.mermaid-diagram`) |
| Shared markdown (if Mermaid in markdown) | `components/chat/SharedMarkdown.tsx`, `components/assistant-ui/markdown-text.tsx` (only if Plan uses markdown-rendered Mermaid elsewhere) |

---

## 4. Dependencies & Order

- **#1 (Mermaid):** Can be done in parallel; requires reference screenshot for padding/readability.
- **#2 (Single Code Gen button):** Code page only; can be done in parallel.
- **#3 (Remove file names):** Code page only; small, can be batched with #2.
- **#4 (Remove four sections):** Plan page only; clear removal of four blocks.
- **#5 (Slices on Plan page):** Plan page; add Slices list/sidebar; design may be needed for layout (e.g. sidebar vs top list).

No design blocker is called out for these items; if "refer to Jayesh screenshot" or slice layout needs formal design, add Pritha/Aditi as in the other scope docs.

---

## 5. Assumptions

- Tuesday is the target due date for the above scope.
- "Slices" on Plan page means a visible slice list/navigation (not backend slice API changes).
- Mermaid change is limited to padding/readability and does not require new diagram types or API changes.
- Removing the four sections is UI-only unless product confirms backend/contract changes.

---

## 6. Out of Scope (for this doc)

- Backend or API changes for plan/spec/codegen.
- New Plan or Code page features beyond the five sub-tasks.
- Broader Mermaid syntax or library upgrades beyond padding/readability.

---

*Document created for planning. Estimates are effort-only (no code in this doc). Update when reference screenshot or design for Mermaid/slices is available.*
