"""Entry point for `python -m sophia`."""

import sys

if "--sse" in sys.argv:
    from sophia.server import main_sse

    main_sse()
else:
    from sophia.server import main

    main()
