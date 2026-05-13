# Project documentation index

Everything in this folder is human-readable narrative and assets for the
project. It is **not** imported by any code in `src/`, `scripts/`, or `tests/`.

```
docs/
  README.md                  # this file
  development_log.md         # running engineering journal across all weeks
  todo.md                    # open follow-ups (was the bare top-level TODO file)
  presentation_slides.md     # final-presentation copy-paste pack for Google Slides
  presentation_images/       # PNGs extracted from notebooks; index.csv lists them
    README.md
    index.csv
    <topic-named subfolders>
    _history_lora/           # plot frames recovered from older git revisions
  reports/                   # weekly deliverable reports (week04..week16)
  critiques/                 # weekly self-critiques (week04, week06, week10, week12, week14, week16)
  slides/                    # PDF exports of each draft deck and the lightning talk
  assignments/               # course-provided assignment templates
  llm_exploration/           # weekly logs of AI-assisted development (frozen)
```

The reports and critiques are renamed to `weekNN.md` (zero-padded) so they
sort chronologically. The LLM-exploration logs and the week-4 assignment
template are kept as historical documents and were not rewritten when other
files were renamed.
