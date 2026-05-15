# Support

## Where to ask

| Channel | Use it for | Status |
| --- | --- | --- |
| GitHub issues | Bug reports, feature requests, documentation problems, reproducible installation failures | Ready |
| Public Zulip stream or topic | Open questions, user support, discussions before opening an issue | In setup |
| Streamlit app | No-code exploration and demonstrations | Ready |

The goal is for community questions to be publicly visible so people can learn
from previous answers without needing private access. A Zulip space is a good
fit because topics remain searchable and readable.

## Zulip setup

The docs are now wired for Zulip in the navbar and community pages. To complete
the integration:

1. Create a public `psytrax` stream in the Neuroinformatics Zulip workspace, or
   choose another public workspace for psytrax.
2. Make the stream readable without login if the workspace settings allow it.
3. Replace the temporary workspace-level Zulip URL in `docs/source/conf.py` and
   `docs/source/snippets/connect-with-us.md` with the direct stream URL.
4. Pin a welcome topic that explains how to ask a good question and links back
   to this support page.

## What to include in a support request

- The version of psytrax you installed.
- Your Python version and operating system.
- The model you are fitting.
- The shape and keys of your data dictionary.
- The smallest code snippet or uploaded example that reproduces the problem.
- The full error message if something failed.

Please do not share identifiable human participant data or private lab data in
public issues. Use a synthetic or reduced example whenever possible.
