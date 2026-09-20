# Patterns

Five shapes that cover almost everything people build on a System One model. Each sketch uses
[`examples/arbiter_client.py`](../examples/arbiter_client.py) and is short enough to paste.

```python
from arbiter_client import ArbiterClient, Choice, Noul, Score
arbiter = ArbiterClient()                      # ARBITER_URL, default http://localhost:8010
```

---

## 1. Speculative fan-out

Ask everything you might need, in one request, before you know which answers you will use. Every
question is a row in the same batch: the encoder runs once, the rows are independent, and ten
questions cost about what one costs. Asking three questions in three requests is the slow way to
do one request.

```python
r = arbiter.system_one(ticket, {
    "department":  Choice("Which team owns this?", {"billing": "payments", "tech": "bugs"}),
    "urgency":     Score("How fast must we reply?", ["no rush", "this week", "today", "now"]),
    "refund":      Noul("The customer is asking for their money back."),
    "churn":       Noul("The customer is threatening to cancel."),
    "frustration": Score("How angry are they?", ["calm", "annoyed", "angry", "furious"]),
})
print(r.choice("department"), r.score("urgency"), r.noul("churn"), r.latency_ms)
```

The cost of an answer you end up not using is one row in a batch. The cost of a second round
trip because you did not ask is a whole request.

---

## 2. Confidence-gated routing

Never `if p > 0.5`. Pick a threshold for acting, a threshold for refusing, and leave a band
between them where a human is asked. Make the two asymmetric in proportion to what each kind of
mistake costs -- a false block on a shell command costs one confirmation, a false allow can cost
the repository.

```python
AUTO, REFUSE = 0.85, 0.30                # code owns these, not the model

p = r.probabilities("department")[r.choice("department")]
if p >= AUTO:
    queue(r.choice("department"))                       # act
elif p < REFUSE:
    queue("triage")                                     # refuse to guess
else:
    ask_human("department %s at p=%.2f" % (r.choice("department"), p))
```

The base checkpoints are calibrated but ship slightly over-confident, so a 0.55 is not the
coin-flip it looks like. Say the number in whatever you print: it lets someone argue with the
threshold rather than with you.

---

## 3. Composite scoring

One question per risk, combined in code. Four separate nouls beat one "how risky is this"
because each has its own threshold and its own consequence, and because you can see which one
fired.

```python
r = arbiter.system_one(diff, {
    "credentials": Noul("This change exposes production credentials."),
    "migration":   Noul("This change contains a database migration."),
    "infra":       Noul("This change alters infrastructure other teams depend on."),
    "rollback":    Noul("This change would be hard to roll back."),
    "blast":       Score("How far does the damage reach?", ["one file", "one service",
                                                            "several services", "everyone"]),
})
if r.noul("credentials") >= 0.65 or r.score("blast") >= 2.6:
    block()
elif any(r.noul(q) >= 0.45 for q in ("migration", "infra", "rollback")):
    request_review()
```

Resist the temptation to add the numbers into a single score. A weighted sum hides which signal
fired, and the weights are a second set of magic numbers nobody can justify.

---

## 4. Intent routing with a hierarchy

`choice` takes at most a dozen options here: the option descriptions share a fixed token budget
with the state, so a long list is less accurate, not just slower. For a large taxonomy, classify
coarsely and then classify again inside the winner. Two forward passes still beat one LLM call.

```python
AREAS = {"billing": "payments and invoices", "product": "the app itself",
         "account": "login and settings", "other": "none of these"}
BILLING = {"refund": "wants money back", "invoice": "wants a document",
           "pricing": "asking what things cost", "failed_payment": "a charge did not go through"}

area = arbiter.system_one(message, {"a": Choice("Which area?", AREAS)}).choice("a")
if area == "billing":
    intent = arbiter.system_one(message, {"i": Choice("Which billing intent?", BILLING)}).choice("i")
```

---

## 5. Cascade with the LLM next door

Let the encoder decide whether the small model is enough. All three Laya checkpoints together
are 1.16B parameters and leave the GPU almost entirely to whatever else is on it, so the "fast"
model can be an LLM served on the same box -- and the same encoder that routed the request can
check the answer that comes back.

```python
r = arbiter.system_one(request, {
    "complexity": Score("How much reasoning does this take?",
                        ["a lookup", "a short answer", "multi-step", "open-ended"]),
    "tools":      Noul("Answering this requires calling tools."),
})
if r.score("complexity") <= 0.9 and r.noul("tools") < 0.4:
    answer = fast_model(request)                        # the LLM next door
    ok = arbiter.system_one({"request": request, "answer": answer},
                         {"good": Noul("This answer is correct and complete.")}).noul("good")
    answer = answer if ok >= 0.7 else powerful_model(request)
else:
    answer = powerful_model(request)
```

The check is the part that makes the cascade safe to run: without it you are betting the whole
quality of the system on one 0-3 score, and an ordinal score is the weakest of the three
primitives.

---

## Choosing a primitive

| You want | Use | Note |
|---|---|---|
| "is X true of this?" | `Noul` | the strongest of the three; phrase it as a statement, not a question |
| "which one of these?" | `Choice` | keep it to ~12 options; split hierarchically past that |
| "how much / how bad?" | `Score` | weakest; if there is a boundary you care about, make it a `Noul` |

`Noul("this change needs a database migration")` beats `Score("how risky is this change")` plus
a cutoff, every time. The score makes the model agree with you about what the middle means; the
noul only asks about the thing you actually care about.

## Writing the state

Pass an object when the thing has fields -- `{"from": ..., "subject": ..., "body": ...}` reads
better than the same text flattened into a sentence, because the sender is visibly a sender. The
read window is fixed and fairly short, so put what matters first: a 900-line diff should be
preceded by the description and the file list.

For a list of items under one shared state, name the item in each question
(`"Passage p3 contains ..."`). Identical wording means identical rows, and identical rows
return identical probabilities.
