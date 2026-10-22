# Contributing

## Shape of a change

1. **Discuss** the problem before the code. A change that starts as a diff and
   ends as a discussion costs two rewrites.
2. **Plan** it under `docs/plans/<nnnn>-<name>.md` against
   [`docs/plans/TEMPLATE.md`](docs/plans/TEMPLATE.md). A plan that changes a
   public contract names its owner spec under Related Files
   ([policy](docs/policies/project-policy.json)).
3. **Implement** milestone by milestone. One commit per milestone, message
   prefixed with the milestone tag (`M0:`, `M1:`).
4. **Prove** it: `uv run pytest`, `uv run ruff check .`, `uv run ty check`. A
   milestone is not done while any of the three is red.

## Titles and branches

- Branch: `task/<short-name>`.
- Commit and pull request titles: `<type>: <what changed>` where type is one of
  `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `meta`.
- The body says what the change does, what it does not, and how it was verified.
  A body that restates the title is not a body.

## What a review looks for

- Does the change state a contract it changes, and does the spec move with it?
- Is a refusal a refusal, or a fallback wearing one? Nothing here guesses a
  device, a dtype, a tolerance, or an extent.
- Does every new number a report can print carry the path it was read from?
- Is a test asserting behaviour, or asserting that the source still looks a
  certain way?
