# gui

This is a C+ project. C+ is a young language, so **do not write it from
memory** — the toolchain answers every question about it, offline and
version-matched to this project.

## Before you write any C+

Run `cpc skill`. It prints the language reference, and inside a project
it also prints the reference of every dependency that ships one — facet
contributes several hundred lines about its retained, non-reactive model
and the mistakes that compile anyway.

There is deliberately no SKILL.md checked in beside this file. A copy
drifts from the compiler that wrote it; `cpc skill` cannot, because it
IS the compiler answering — and it is the only form that also carries
your dependencies' references.

`cpc skill --lang-only` is the language alone, if that is all you need.

## When the compiler says no

Run `cpc explain <CODE>` before you guess. Every diagnostic code has a
cause, a fix and a worked example behind it — `cpc explain E0613` is
faster and more reliable than inferring from the message.

## Navigating this code

**Do not grep for definitions.** C+ has no dynamic dispatch, so every
call to a named function resolves and the graph's answer is COMPLETE —
which grep's never is:

```
cpc query definition <symbol>     where is it
cpc query references <symbol>     everywhere it is used
cpc query callers <symbol>        who calls it
cpc query symbols <file>          the outline of a file
cpc query scope-at <file:line:col> what you can type right there
cpc query complete <file:line:col> ...and what fits after a `.` or `::`
```

The same graph is available as MCP tools — see `.mcp.json`, which points
at `cpc mcp`. Prefer either over reading files to find things. Each
`cpc query` rebuilds the whole graph (~seconds on a large project) and
throws it away; the MCP server builds once and answers in microseconds,
so use it for anything more than a single lookup.

## Building

```
cpc build          compile and link
cpc test           run the tests
cpc fmt            canonical formatting (no arg = this project)
```

## Driving the running app

This app is an ACI: while it runs it serves MCP, and you can read its
UI and act on it. `src/app.cplus` is where that is turned on.

**Find it.** The address is derived from the app id and its pid, so a
running instance writes `/tmp/mcp-gui-<pid>.json` saying where
it landed:

```
cat /tmp/mcp-gui-*.json
```

A pid whose process is gone is a leftover — check with `kill -0 <pid>`.
If you launched the app yourself you already know the pid, so you can
skip the file: the port is `9000 + pid % 1000`.

**Talk to it.** Plain JSON-RPC over POST, no bridge:

```
curl -s -X POST http://127.0.0.1:<port>/ \
     -d '{"jsonrpc":"2.0","id":1,"method":"describe_ui"}'
```

`tools/list` names every verb, and the startup line says how many
there are. The core eleven are `describe_ui`, `click`, `set_text`,
`hit_test`, `set_caret`, `read_text`, `read_runs`, `invoke_menu`,
`scroll_to`, `poll_event` and `activity`. Fourteen more see the whole
tree rather than just what is exposed, and can write any property on
it: `describe_tree`, `inspect`, `set`, `set_many`, `reset`, `nudge`,
`insert`, `remove`, `reparent`, `undo`, `highlight`,
`clear_highlight`, `vocabulary` and `journal`. They come with the
surface — there is nothing extra for the app to call. Ask
`vocabulary` what a property takes before you `set` one.

`activity` is the record of what has been DONE through the surface —
useful when a person is supervising you, and when you want to check
what you already tried.

**This is how you test a UI change.** `describe_ui` answers a flat node
list and each `id` is the `key:` written in the code. Click, describe
again, and read the change — that is evidence, in a way "it should work
now" is not.

Two things worth knowing before you are confused by them:

- **You have no hands.** There is no drag, pinch or swipe verb and
  there will not be one. An affordance only a gesture can reach is a
  bug in the app — it is unreachable for anyone driving by voice too.
  Fix the click path; do not look for a gesture verb.
- **You may be refused once.** If the app wired `agent_consent`, your
  first request is refused while a dialog asks the user. The error
  says whether to retry — `consent pending` means come back,
  `consent denied` means the user said no.
