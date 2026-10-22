---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
---

<!-- Read this before writing any of the plan below.

     Pass 1 -- audit the repository and probe what is not yet known. Write
     `### Current state`, and nothing else yet.

     Pass 2 -- write each milestone's `##### Delivered` shape. Then stop and show
     the author those shapes. No prose exists yet, so a shape that is not what the
     author meant costs ten lines to redirect rather than a document to rewrite.

     Pass 3 -- write the rest.

     Append to `### Decisions` as each question is settled. Never edit an accepted
     record; append one that supersedes it.

     Settle with the author anything an implementer would otherwise decide alone:
     the delivered code and the way it is accepted alike.

     Use only the headings below. -->

# [TYPE][component] <short description>

## Description
<!-- The problem and the intended result. Only what the reader needs before the
     current state. -->

### Current state
<!-- The mechanism the code implements today: what is there, and what is missing.
     No design and no proposal -- the gap is the finding, not the fix.

     One bullet per claim, each citing a path in backticks. A claim with nothing
     to point at is the one that turns out to be wrong. -->
- `src/trainer_rsi/<file>.py` <what it does today>

### Decisions
<!-- Append-only. One settled question per record, never edited once accepted:
     when the answer changes, append a record that supersedes it.
     `None.` when the plan settled nothing an implementer would otherwise choose. -->
- D1 <question> -- <choice>, because <reason>.

## Milestones

### Milestone M0: <name>

#### Depends
- None

#### Target State Design

##### Delivered
<!-- The surface this milestone designs and its output, in delivered form. At
     least one block MUST be a ```diff headed by `# <path>`, because plain code
     states the end state but not which lines are new. -->

```diff
# src/trainer_rsi/<file>.py
+ <added line>
```

##### Accepted by
<!-- Behaviour, not a test obligation. -->

#### Related Files
- `docs/spec/<owner>.md` (or none, when no public contract changes)

#### Rejected
<!-- Approaches considered and why they lost. -->
