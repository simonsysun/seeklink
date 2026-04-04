"""Entry point for `python -m seeklink`."""

import sys

if "--sse" in sys.argv:
    from seeklink.server import main_sse

    main_sse()
else:
    from seeklink.server import main

    main()
