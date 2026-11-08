# Notes for Claude

## Pythonista: never use SystemExit

Scripts in this repository are run in Pythonista on iOS, where `SystemExit`
crashes the app instead of just ending the script. So:

- Never `raise SystemExit(...)`, `sys.exit()`, `exit()` or `quit()`, and don't
  write `if __name__ == '__main__': sys.exit(main())`.
- Handle errors gracefully: raise an ordinary exception (e.g. `hf_run.py`'s
  `RunError`), catch it at the script's entry point, print a clear message
  saying what went wrong and what to do, and return.
- For an early stop in a script, `print(...)` and `return` from `main()`.
- Code that may raise `SystemExit` itself (argparse on `--help` or a bad
  option) is caught at the entry point.
