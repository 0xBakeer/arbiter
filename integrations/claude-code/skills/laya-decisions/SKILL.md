---
name: laya-decisions
description: Use the laya MCP tools (laya_check, laya_classify, laya_score, laya_gate, laya_decide) to make fast, typed decisions instead of reasoning about routine judgements. Use when triaging, classifying, ranking, filtering, or deciding whether an action is safe, and when the same judgement has to be made over many items.
---

# Deciding with Laya instead of thinking about it

Laya is a small encoder served locally that answers typed questions about a piece of state in
one forward pass: a probability, a label with its distribution, or a position on a scale. It
does not write prose and it cannot be argued with. That makes it the right tool for judgements
you would otherwise make yourself, one careful paragraph at a time.

Reach for it when:

- the same judgement has to be made over many items (passages, files, tickets, alerts, diffs);
- the judgement is routine but you want it consistent, not improvised;
- something is about to happen that could be destructive, and you want a second opinion before
  it does.

Do not reach for it when the answer is a piece of writing, a plan, or anything that needs
knowledge of the world beyond what you put in the state. It reads the state you give it and
nothing else.

## Pick the primitive

| Tool | Answers | Use it for |
|---|---|---|
| `laya_check` | one probability, 0-1 | "is this true of the state?" -- the strongest primitive |
| `laya_classify` | one of up to 12 labels, plus the distribution | routing, intent, category, owner |
| `laya_score` | a float on a 2-10 level scale | severity, urgency, size, blast radius |
| `laya_gate` | allow / confirm / block plus five signals | before running a destructive action |
| `laya_decide` | everything above, batched | more than one question about one state |

**Default to `laya_decide`.** Every question in a call is a row in the same batch, so ten
questions cost roughly what one costs. Ask everything you might want to know at once, then let
your own logic combine the answers. Two sequential calls are the slow way to do one call.

**Prefer a check to a score.** An ordinal scale is the weakest of the three: the model has to
agree with you about what the middle means. If there is a boundary you actually care about,
phrase the boundary as a statement and use `laya_check`: "this change needs a database
migration" beats a 0-3 "how risky is this change" plus a cutoff.

## Write the state like a record, not a paragraph

Pass an object when the thing has fields. `{"from": ..., "subject": ..., "body": ...}` reads
better than the same text flattened into a sentence, because the sender is visibly a sender.
Put what matters first: the read window is fixed and fairly short, so a 900-line diff should be
preceded by the description and the file list, which are the parts certain to be read.

For a list -- passages, files, candidates -- put the whole list in the state under stable keys
and ask one question per key, naming the key in the instruction ("Passage p3 contains ..."). If
every question is worded identically the rows are identical and so are the answers.

## Write the question as a statement, not a request

Good: `"This message is a phishing attempt: it impersonates a sender and pushes the reader to
click a link or hand over credentials."`

Bad: `"Is this phishing?"`

Add `true_desc` and `false_desc` when the boundary is genuinely unclear -- they are what stops
the model from using its own definition of a word like "urgent" or "destructive".

## Respect the option budget

`laya_classify` takes at most 12 options here, and the tool will refuse more. Option
descriptions share a fixed token budget with the state, so a long list does not merely cost
more, it gets less accurate. For a large taxonomy, classify coarsely first and then classify
again inside the winning group. Two forward passes still beat one model call.

## The thresholds are yours

The model reports probabilities. It does not know what you intend to do with them, and it ships
slightly over-confident, so a 0.55 is not the coin-flip it looks like. Never write `if p > 0.5:
do the thing`. Instead:

1. pick a threshold for acting automatically;
2. pick a lower one for refusing;
3. **leave a band in the middle where you ask the user.**

Make the two thresholds asymmetric in proportion to what a mistake costs. A false "block" on a
shell command costs one confirmation; a false "allow" can cost the repository. State the numbers
you used when you report a decision, so the user can argue with the threshold rather than with
you.

## Before you run something destructive

Call `laya_gate` with the command and a sentence of context. It returns `allow`, `confirm` or
`block` plus the five signals behind it. Treat `confirm` as "ask the user, quoting the signal",
not as "probably fine". If the server is not reachable, the gate raises an error: that is not
permission to proceed, it is a missing opinion.
