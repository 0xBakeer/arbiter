# README diagrams

Two SVGs, drawn by hand, no external fonts. Text uses `currentColor`; strokes and accents use
CSS variables set inside each file with a `prefers-color-scheme: dark` override, so they read
correctly on a light or dark page whether embedded inline or loaded through `<img>`.

| File | Shows |
|---|---|
| `one-call-typed-answers.svg` | A state plus typed questions go into the encoder, one forward pass, and every question comes back as a typed answer with calibrated probabilities. 900 x 300. |
| `cascade-next-to-an-llm.svg` | Where Laya sits: it decides in milliseconds, plain code handles the deterministic majority, a local LLM handles generation, a frontier model only sees the escalated few. 900 x 380. |

## Embedding in the README

Plain Markdown, which GitHub renders through its image proxy:

```markdown
![One call, many typed answers](docs/assets/one-call-typed-answers.svg)

![The cascade next to an LLM](docs/assets/cascade-next-to-an-llm.svg)
```

The dark override inside each SVG follows the operating system's colour scheme, which is what
GitHub's "sync with system" theme uses. The same file also works in an HTML `<img>` when the
README needs a fixed width or centring:

```html
<p align="center">
  <img src="docs/assets/one-call-typed-answers.svg" alt="One call, many typed answers" width="900">
</p>
```

Inline in a page you control, paste the `<svg>` element directly; its styles are scoped to the
`.arbiter-diagram` class and override nothing else. To tint the accents to a site palette, set
`--accent` and `--accent2` on the `.arbiter-diagram` element.
