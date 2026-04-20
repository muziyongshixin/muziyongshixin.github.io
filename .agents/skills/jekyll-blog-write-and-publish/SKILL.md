---
name: jekyll-blog-write-and-publish
description: Draft researched blog posts for Jekyll or GitHub Pages sites, add lightweight visuals, validate post assets, and publish the post safely with git. Use when the user wants a new post created under `_posts`, an existing post rewritten for publication, diagrams added under `images/posts`, or the finished blog changes committed and pushed.
---

# Jekyll Blog Write And Publish

## Overview

Use this for personal sites or blogs where content is stored as Jekyll markdown files and published through git. The workflow covers repo convention discovery, research-backed drafting, optional SVG diagrams, lightweight validation, and safe publishing.

## Workflow

### 1. Inspect repo conventions first

- Read `README.md` and 2-3 existing posts before writing.
- Copy the repo's front matter style, permalink pattern, tag style, and image embedding style.
- Identify where posts and images live. In this repo they are usually `_posts/` and `images/posts/`.

### 2. Research before drafting

- If the topic is technical, recent, or citation-sensitive, browse first.
- Prefer primary sources: papers, official docs, OpenReview, arXiv, CVF, PMLR, official project pages.
- Reshape raw user notes into a clean narrative instead of copying the note structure directly.
- Preserve equations only when they improve understanding.

### 3. Draft the post for publication

- Create posts with a date-prefixed filename such as `YYYY-MM-DD-blog-post-topic.md`.
- Keep the opening short: what the article explains, why it matters, and how it is organized.
- Prefer a reader-friendly flow:
  1. problem setup
  2. historical context
  3. core mechanism
  4. tradeoffs
  5. later extensions
  6. references
- If the user gives dense notes, remove repetition and convert note fragments into blog prose.

### 4. Add visuals only when they help

- Prefer lightweight SVG diagrams committed into `images/posts/`.
- Good candidates:
  - timeline of method evolution
  - side-by-side method comparison
  - workflow or information-flow diagram
- Keep visuals self-contained and dependency-free.
- In this repo, embed images with:

```html
<br />
<img align="center" width="1000" src="{{ site.url }}/images/posts/file.svg" alt="description">
<br />
```

### 5. Validate lightly but explicitly

- Confirm front matter parses.
- Confirm every referenced local image exists.
- If the environment lacks `jekyll`, say so and do lightweight checks instead of pretending the build passed.
- Good lightweight checks:

```bash
ruby -e 'c=File.read(ARGV[0]); parts=c.split(/^---\s*$/); require "yaml"; p YAML.load(parts[1])' path/to/post.md
git status --short --branch
```

### 6. Publish safely

- Stage only the post and related assets.
- Never stage `.DS_Store` or unrelated files by default.
- Inspect branch state before pushing. If the current branch is already ahead, warn that earlier local commits will also be pushed.
- If HTTPS push fails but SSH auth works, push via explicit SSH remote rather than rewriting `origin`.
- Use clear commit messages such as `Add diffusion CFG explainer post`.

## Repo-specific notes for this project

- Posts live in `_posts/`.
- Diagrams for posts fit naturally in `images/posts/`.
- Existing posts often use `toc: true`.
- The repo may contain local `.DS_Store` files; keep them untracked.

## Resources (optional)

No extra resources are required for the first version of this skill.
