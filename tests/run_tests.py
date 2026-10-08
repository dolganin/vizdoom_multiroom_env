#!/usr/bin/env python
"""Dependency-free test runner (the tests are also valid pytest tests).

    python tests/run_tests.py                 # everything, 1 process
    python tests/run_tests.py -j 2            # 2 worker processes
    python tests/run_tests.py generator       # only tests whose id contains "generator"
    python tests/run_tests.py --list

A test is any function ``test_*`` in a module ``tests/test_*.py``; it passes if
it returns without raising.  Each worker process limits itself to 2 threads.
"""
from __future__ import annotations

import argparse
import importlib
import io
import multiprocessing
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def _setup() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import helpers  # noqa: F401  (puts src/ on sys.path)


def collect() -> list:
    _setup()
    ids = []
    for fname in sorted(os.listdir(HERE)):
        if fname.startswith("test_") and fname.endswith(".py"):
            mod = importlib.import_module(fname[:-3])
            ids += ["%s::%s" % (fname[:-3], name) for name, obj in vars(mod).items()
                    if name.startswith("test_") and callable(obj)]
    return ids


def run_one(test_id: str):
    _setup()
    from contextlib import redirect_stdout
    mod_name, fn_name = test_id.split("::")
    buf = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(buf):
            getattr(importlib.import_module(mod_name), fn_name)()
        return test_id, True, time.time() - t0, buf.getvalue()
    except BaseException:            # noqa: B902 - report everything, including SystemExit
        return test_id, False, time.time() - t0, buf.getvalue() + traceback.format_exc()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("patterns", nargs="*", help="run only tests whose id contains a pattern")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="worker processes (default 1)")
    ap.add_argument("--list", action="store_true", help="list the test ids and exit")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not show the output of "
                                                               "passing tests")
    args = ap.parse_args(argv)
    ids = [i for i in collect() if not args.patterns or any(p in i for p in args.patterns)]
    if args.list:
        print("\n".join(ids))
        return 0
    t0 = time.time()
    failed = []

    def report(res):
        test_id, ok, seconds, output = res
        print("%s %-62s %7.1f s" % ("PASS" if ok else "FAIL", test_id, seconds), flush=True)
        if output.strip() and (not ok or not args.quiet):
            print("    " + output.rstrip().replace("\n", "\n    "), flush=True)
        if not ok:
            failed.append(test_id)

    if args.jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(run_one, ids, chunksize=1):
                report(res)
    else:
        for test_id in ids:
            report(run_one(test_id))
    print("\n%d tests, %d passed, %d failed in %.1f s (%d process%s)" % (
        len(ids), len(ids) - len(failed), len(failed), time.time() - t0, args.jobs,
        "" if args.jobs == 1 else "es"))
    for test_id in failed:
        print("FAILED", test_id)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
